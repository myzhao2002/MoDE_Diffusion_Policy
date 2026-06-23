#!/bin/bash
# 扫"大脑时延 vs 恢复成功率"曲线: 固定 瞬间启发式决策 + 立刻打断, 只变人为时延。
# 每个 (时延, 初始化) 跑一次, 记成功/失败。用于判断: 时延是否真的单调拖垮恢复(还是单次噪声)。
cd /root/autodl-tmp/MoDE_Diffusion_Policy || exit 1
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD/LIBERO:$PWD
PY=/root/miniconda3/envs/lerobot/bin/python
TF=/root/autodl-tmp/MoDE_Diffusion_Policy/logs/runs/2026-06-22/13-59-34
CK=/root/autodl-tmp/eval_emafix.ckpt
OUT=/root/autodl-tmp/latency_sweep.txt
echo "# delay_s init result" > $OUT
for d in 0 1 2 3 4; do
  for i in 0 1 2 3 4; do
    LINE=$(timeout 400 $PY scripts/closed_loop.py --train_folder $TF --checkpoint $CK \
      --inject open_gripper --inject_step 95 --inject_len 25 --max_steps 500 \
      --recover --brain heuristic --interrupt --brain_delay_s $d --init $i 2>&1 \
      | grep -a "success=" | tail -1)
    echo "d=$d i=$i $LINE" >> $OUT
    echo "[sweep] d=$d i=$i -> $LINE"
  done
done
echo "ALLDONE" >> $OUT
