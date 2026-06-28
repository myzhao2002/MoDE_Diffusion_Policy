#!/bin/bash
# 下载 VyoJ/calvin-ABCD-D-subsets 的训练子集(hf-mirror + aria2 多线程),一次一个、解压后删 zip。
# 用法: bash dl_calvin_subsets.sh [起始编号] [结束编号]
#   默认 0..5 共 6 个子集 (~60万帧, ~170G 解压)。
# 在 screen 里跑: screen -S dlcalvin  然后 bash scripts/dl_calvin_subsets.sh 0 5
# 断点续传: aria2c -c,中断后重跑同命令即可接着下。
set -u

START=${1:-0}
END=${2:-5}

REPO="VyoJ/calvin-ABCD-D-subsets"
ENDPOINT="https://hf-mirror.com"
STAGE="/root/autodl-tmp/CALVIN-datasets/_subsets_raw"   # zip 暂存 + 解压区
ZIPDIR="$STAGE/zips"
mkdir -p "$ZIPDIR"

echo "[dl] subsets $START..$END  endpoint=$ENDPOINT  stage=$STAGE"
for i in $(seq "$START" "$END"); do
  n=$(printf "%03d" "$i")
  fn="subset_training_${n}.zip"
  url="$ENDPOINT/datasets/$REPO/resolve/main/training/$fn"
  outzip="$ZIPDIR/$fn"
  outdir="$STAGE/subset_training_${n}"

  if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
    echo "[dl] $n already extracted -> skip"; continue
  fi

  echo "=== [$n] downloading $fn ==="
  # xet 大文件: 串行下载, -x16 -s16, -c 续传。--max-tries/--retry-wait 让 aria2 在
  # xet 签名URL过期(403)时重新解析 hf-mirror URL 拿新签名; 再叠加脚本级 3 次重试兜底。
  ok=0
  for attempt in 1 2 3; do
    aria2c -x 16 -s 16 -c --max-tries=10 --retry-wait=15 \
           --timeout=60 --connect-timeout=30 --max-file-not-found=5 \
           --file-allocation=none --summary-interval=20 \
           -d "$ZIPDIR" -o "$fn" "$url" && { ok=1; break; }
    echo "[dl][WARN] $n 第 $attempt/3 次失败, 15s 后重试(aria2 -c 续传)"; sleep 15
  done
  if [ "$ok" -ne 1 ]; then echo "[dl][ERR] $n 三次仍失败, 终止(重跑本脚本可续传)"; exit 1; fi

  echo "=== [$n] unzip -> $outdir ==="
  mkdir -p "$outdir"
  unzip -q -o "$outzip" -d "$outdir"
  if [ $? -ne 0 ]; then echo "[dl][ERR] $n unzip failed"; exit 1; fi

  echo "=== [$n] rm zip ==="
  rm -f "$outzip"
  df -h /root/autodl-tmp | tail -1
done

echo "[dl] 全部完成。子集解压在 $STAGE 。下一步跑 merge_calvin_subsets.py 合并。"
ls -d "$STAGE"/subset_training_* 2>/dev/null
