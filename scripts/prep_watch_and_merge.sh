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
"$PY" scripts/merge_calvin_subsets.py --stage "$STAGE" --out "$OUT/training" || { echo "[watch][ERR] merge 失败"; exit 1; }

if [ ! -e "$OUT/validation" ]; then
  ln -s "$VAL_SRC" "$OUT/validation" && echo "[watch] 软链 validation -> $VAL_SRC"
fi

echo "[watch] ===== 数据就绪 $OUT ($(date)) ====="
ls "$OUT"
echo "episodes: $(ls "$OUT"/training/episode_*.npz 2>/dev/null | wc -l)"
df -h /root/autodl-tmp | tail -1
echo "[watch] 下一步(切到 GPU 后): nohup bash scripts/run_calvin_xattn_ab.sh > /root/autodl-tmp/calvin_ab.log 2>&1 &"
