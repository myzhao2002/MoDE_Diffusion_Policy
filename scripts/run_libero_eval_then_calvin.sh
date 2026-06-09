#!/bin/bash
# 一次性编排:① LIBERO exp4 全集评测 → ② CALVIN exp4 微调(带闭环 rollout 评测)→ ③ 可选关机
# 用法:  bash scripts/run_libero_eval_then_calvin.sh            # 跑完不关机
#         bash scripts/run_libero_eval_then_calvin.sh shutdown   # 跑完自动关机
#
# 前置(本机已就绪):calvin_env 已 pip install -e、pybullet/numpy-quaternion 已装、
#   calvin_env play_table_env.py:72 已打补丁、calvin_vyoj/{training,validation}/.hydra/merged_config.yaml 已生成。
set -u
cd /root/autodl-tmp/MoDE_Diffusion_Policy
source /etc/network_turbo 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/MoDE_Diffusion_Policy/LIBERO:/root/autodl-tmp/MoDE_Diffusion_Policy
TS=$(date +%Y%m%d_%H%M%S)

echo "==================== [1/2] LIBERO exp4 full eval ($(date)) ===================="
python mode/evaluation/mode_evaluate_libero.py \
  train_folder=/root/autodl-tmp/MoDE_Diffusion_Policy/logs/runs/2026-06-08/17-03-08 \
  checkpoint=/root/autodl-tmp/MoDE_ckpts/last-v6.ckpt \
  plan_file=/root/autodl-tmp/LIBERO-datasets/plans_libero_all_tasks.jsonl \
  log_wandb=False \
  2>&1 | tee /root/autodl-tmp/eval_libero_exp4_${TS}.log
echo "[1/2] LIBERO eval done ($(date))"

echo "==================== [2/2] CALVIN exp4 finetune + closed-loop eval ($(date)) ===================="
python mode/training_calvin.py \
  root_data_dir=/root/autodl-tmp/CALVIN-datasets/calvin_vyoj \
  lang_folder=lang_filtered \
  plan_file=/root/autodl-tmp/CALVIN-datasets/plans_calvin.jsonl \
  use_extracted_rel_actions=False \
  model.start_from_pretrained=True model.ckpt_path=/root/autodl-tmp/MoDE_Pretrained \
  model.use_resnet_xattn=True \
  callbacks.rollout_lh.skip_epochs=19 callbacks.rollout_lh.rollout_freq=1 \
  callbacks.rollout_lh.num_sequences=200 \
  devices=1 batch_size=32 trainer.strategy=auto trainer.sync_batchnorm=False \
  max_epochs=20 \
  logger.entity=null \
  hydra.run.dir=/root/autodl-tmp/MoDE_ckpts/calvin_exp4_${TS} \
  2>&1 | tee /root/autodl-tmp/train_calvin_exp4_${TS}.log
echo "[2/2] CALVIN train+eval done ($(date))"

echo "==================== ALL DONE ($(date)) ===================="
echo "LIBERO eval log:  /root/autodl-tmp/eval_libero_exp4_${TS}.log"
echo "CALVIN train log: /root/autodl-tmp/train_calvin_exp4_${TS}.log"
echo "CALVIN outputs:   /root/autodl-tmp/MoDE_ckpts/calvin_exp4_${TS}/"

if [ "${1:-}" = "shutdown" ]; then
  echo "训练评测全部完成,30 秒后关机..."
  sleep 30
  /usr/bin/shutdown -h now
fi
