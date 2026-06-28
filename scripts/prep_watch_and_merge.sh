#!/bin/bash
# 等 6 子集下载完成(dl_calvin.log 出现 "全部完成") -> 自动 merge -> 软链 validation。
# 无卡模式下挂着即可, 下载完会自动把数据备好。
#   nohup bash scripts/prep_watch_and_merge.sh > /root/autodl-tmp/prep_watch.log 2>&1 &
set -u
LOG="${DL_LOG:-/root/autodl-tmp/dl_calvin.log}"
STAGE="${STAGE:-/root/autodl-tmp/CALVIN-datasets/_subsets_raw}"
OUT="${OUT:-/root/autodl-tmp/CALVIN-datasets/calvin_abcd_6sub}"
VAL_SRC="${VAL_SRC:-/root/autodl-tmp/CALVIN-datasets/calvin_vyoj/validation}"
PY="${PY:-/root/miniconda3/envs/lerobot/bin/python}"
cd /root/autodl-tmp/MoDE_Diffusion_Policy || exit 1

echo "[watch] 等待下载完成 ($(date)) ; log=$LOG"
done_ok=0
for i in $(seq 1 480); do            # 最多等 8h (60s × 480)
  if grep -q "全部完成" "$LOG" 2>/dev/null; then done_ok=1; echo "[watch] 检测到下载完成 ($(date))"; break; fi
  if ! pgrep -f dl_calvin_subsets >/dev/null && ! pgrep -f aria2c >/dev/null; then
    sleep 20
    if grep -q "全部完成" "$LOG" 2>/dev/null; then done_ok=1; break; fi
    echo "[watch][ERR] 下载进程已退出但未见完成标记 -> 可能失败, 放弃 merge。查 $LOG"; exit 1
  fi
  sleep 60
done
if [ "$done_ok" -ne 1 ]; then echo "[watch][ERR] 超时未完成"; exit 1; fi

echo "[watch] 开始 merge ($(date))"
"$PY" scripts/merge_calvin_subsets.py --stage "$STAGE" --out "$OUT/training" || { echo "[watch][ERR] merge 失败 -> 保留实例不关机, 早上看 prep_watch.log"; exit 1; }

if [ ! -e "$OUT/validation" ]; then
  ln -s "$VAL_SRC" "$OUT/validation" && echo "[watch] 软链 validation -> $VAL_SRC"
fi

# ---- 合并自检(过关才允许自动关机, 防止半成品数据被当成功) ----
NEP=$(find "$OUT/training" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l)
MIN_EP="${MIN_EP:-400000}"   # 6 子集 ≈ 60万帧; 低于 40万视为合并异常
LANG_OK=0
"$PY" -c "import numpy as np; d=np.load('$OUT/training/lang_annotations/auto_lang_ann.npy',allow_pickle=True).item(); n=len(d['info']['indx']); print('[watch] lang 标注数 =',n); exit(0 if n>0 else 1)" && LANG_OK=1
echo "[watch] ===== 合并结果: episodes=$NEP  lang_ok=$LANG_OK  ($(date)) ====="
ls "$OUT"; df -h /root/autodl-tmp | tail -1

SANE=1
if [ "$NEP" -lt "$MIN_EP" ]; then echo "[watch][WARN] episode 数 $NEP < $MIN_EP, 合并可能不完整"; SANE=0; fi
if [ "$LANG_OK" -ne 1 ]; then echo "[watch][WARN] lang 标注异常"; SANE=0; fi

# ---- 自动关机(默认 on-success: 自检通过才关, 省过夜无卡机时; 数据在持久盘不丢) ----
# 早上切 GPU 模式本就要重启, 故关机零损失。失败/不健康则保留实例待查。
SHUTDOWN_AFTER="${SHUTDOWN_AFTER:-on-success}"
if [ "$SANE" -eq 1 ]; then
  echo "[watch] 数据就绪健康。下一步(切 GPU 后): nohup bash scripts/run_calvin_xattn_ab.sh > /root/autodl-tmp/calvin_ab.log 2>&1 &"
  case "$SHUTDOWN_AFTER" in
    on-success|always) echo "[watch] 数据备好 -> 30s 后自动关机省机时 ($(date))"; sleep 30; shutdown ;;
    never) echo "[watch] SHUTDOWN_AFTER=never -> 不关机" ;;
  esac
else
  echo "[watch][ERR] 自检未过 -> 保留实例不关机, 早上排查 prep_watch.log"
  exit 1
fi
