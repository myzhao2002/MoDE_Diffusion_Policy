#!/bin/bash
# =============================================================================
# CALVIN 实验四 三臂消融: baseline(原版MoDE) / v1(加plan) / v2(plan+视觉Q-planKV) + 训完自动关机
#
# 用法(GPU 模式下):
#   nohup bash scripts/run_calvin_xattn_ab.sh > /root/autodl-tmp/calvin_ab.log 2>&1 &
#   tail -f /root/autodl-tmp/calvin_ab.log
#
#   SHUTDOWN_MODE=never bash scripts/run_calvin_xattn_ab.sh   # 不关机(调试)
#   ARM=v2  bash scripts/run_calvin_xattn_ab.sh               # 只跑某臂: baseline | v1 | v2
#   ROOT=/path BATCH_SIZE=48 MAX_EPOCHS=30 bash ...           # 覆盖默认
#
# 前置: 数据已 merge 到 $ROOT/{training,validation};calvin_env 已装好(rollout 需要)。
# 关机守卫: on-success = 三臂全成功才关机;任一失败保留实例调试。
# =============================================================================
set -u

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/MoDE_Diffusion_Policy}"
ROOT="${ROOT:-/root/autodl-tmp/CALVIN-datasets/calvin_abcd_6sub}"
PLAN="${PLAN:-/root/autodl-tmp/CALVIN-datasets/plans_calvin.jsonl}"
PRETRAINED="${PRETRAINED:-/root/autodl-tmp/MoDE_Pretrained}"
SHUTDOWN_MODE="${SHUTDOWN_MODE:-on-success}"   # always | on-success | never
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
ARM="${ARM:-all}"                               # all | baseline | v1 | v2
CKPT_BASE="${CKPT_BASE:-/root/autodl-tmp/MoDE_ckpts_calvin}"

# ---- env ----
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lerobot 2>/dev/null || true
source /etc/network_turbo 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"   # 在线看曲线; 需 wandb 已登录(~/.netrc)。离线: WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR/LIBERO:$PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd $PROJECT_DIR"; exit 1; }
echo "python: $(which python)   GPU:"; nvidia-smi -L 2>/dev/null || echo "  (无卡? 训练需要 GPU)"

# ---- data check ----
if [ ! -d "$ROOT/training" ] || [ -z "$(find "$ROOT/training" -maxdepth 1 -name 'episode_*.npz' -print -quit 2>/dev/null)" ]; then
  echo "FATAL: 训练数据缺失 $ROOT/training"
  echo "       先跑: bash scripts/dl_calvin_subsets.sh 0 5"
  echo "             python scripts/merge_calvin_subsets.py --stage <stage> --out $ROOT/training"
  exit 2
fi
if [ ! -e "$ROOT/validation" ]; then
  echo "FATAL: 缺 $ROOT/validation (rollout 评测需要); 软链 vyoj/validation 过去"
  exit 2
fi

# 公共 hydra 覆盖(与历史 CALVIN exp4 命令一致: lang_filtered / 非extracted / 单卡 / 轻量rollout)
COMMON="root_data_dir=$ROOT lang_folder=lang_filtered plan_file=$PLAN \
 use_extracted_rel_actions=False \
 model.start_from_pretrained=True model.ckpt_path=$PRETRAINED \
 callbacks.rollout_lh.skip_epochs=19 callbacks.rollout_lh.rollout_freq=1 callbacks.rollout_lh.num_sequences=200 \
 devices=1 batch_size=$BATCH_SIZE trainer.strategy=auto trainer.sync_batchnorm=False \
 max_epochs=$MAX_EPOCHS logger.entity=null logger.project=calvin_xattn_ablation"

run_arm () {
  local name="$1"; shift
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local dir="$CKPT_BASE/${name}_${ts}"
  echo "==================== ARM=$name START $(date) ===================="
  echo "  outdir=$dir  extra: $*"
  # shellcheck disable=SC2086
  python mode/training_calvin.py $COMMON "$@" hydra.run.dir="$dir"
  local code=$?
  echo "==================== ARM=$name EXIT code=$code $(date)  (ckpt: $dir/saved_models) ===================="
  return $code
}

CODE=0
# 三臂消融阶梯:
#   baseline = 原版 MoDE, 不加 plan      (use_plan_fusion=False, 无 xattn)
#   v1       = 加 plan(融进 goal)       (use_plan_fusion=True,  无 xattn)
#   v2       = plan + 视觉Q/分步plan-KV   (use_plan_fusion=True,  use_resnet_xattn=True, xattn_visual_query=True)
# baseline->v1 看"plan 有没有用"; v1->v2 看"新视觉交叉注意力在 plan 之上有没有用"。
if [ "$ARM" = "all" ] || [ "$ARM" = "baseline" ]; then
  run_arm baseline model.use_plan_fusion=False model.use_resnet_xattn=False || CODE=$?
fi
if [ "$ARM" = "all" ] || [ "$ARM" = "v1" ]; then
  run_arm v1_plan  model.use_plan_fusion=True  model.use_resnet_xattn=False || CODE=$?
fi
if [ "$ARM" = "all" ] || [ "$ARM" = "v2" ]; then
  run_arm v2_xattn model.use_plan_fusion=True  model.use_resnet_xattn=True model.xattn_visual_query=True || CODE=$?
fi

echo "==================== ALL DONE code=$CODE $(date) ===================="

# ---- shutdown guard (照搬 train_and_shutdown.sh 语义) ----
case "$SHUTDOWN_MODE" in
  always)     echo "[shutdown] always -> 关机"; shutdown ;;
  on-success) if [ "$CODE" -eq 0 ]; then echo "[shutdown] 成功 -> 关机"; shutdown;
              else echo "[shutdown] 失败 code=$CODE -> 保留实例调试"; fi ;;
  never)      echo "[shutdown] never -> 不关机" ;;
  *)          echo "[shutdown] 未知 SHUTDOWN_MODE=$SHUTDOWN_MODE -> 不关机" ;;
esac
exit "$CODE"
