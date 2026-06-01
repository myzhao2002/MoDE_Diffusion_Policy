#!/usr/bin/env bash
# =============================================================================
# train_and_shutdown.sh
#   Train MoDE on LIBERO (with task-plan fusion) and shut the autodl instance
#   down when training finishes. Survives SSH disconnects when launched with
#   nohup. Logs everything to a timestamped file under autodl-tmp.
#
#   训练 MoDE(带 plan fusion),训练结束后自动关机停止计费。配合 nohup 使用,
#   SSH 断开也不会中断;全程日志写到 autodl-tmp 下的带时间戳文件里。
#
# Usage / 用法:
#   nohup bash train_and_shutdown.sh >/dev/null 2>&1 &
#   tail -f /root/autodl-tmp/train_*.log      # 看日志
#
#   # 只在训练成功(退出码 0)时关机,失败则保留实例方便调试:
#   SHUTDOWN_MODE=on-success nohup bash train_and_shutdown.sh >/dev/null 2>&1 &
#
#   # 完全不关机:
#   SHUTDOWN_MODE=never bash train_and_shutdown.sh
#
#   # 任何额外的 hydra 覆盖参数都会原样透传给训练命令,例如:
#   bash train_and_shutdown.sh batch_size=32 trainer.max_epochs=30
# =============================================================================
set -u

# ---- config (override via env vars) / 可用环境变量覆盖 ----------------------
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/MoDE_Diffusion_Policy}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
PLAN_FILE="${PLAN_FILE:-/root/autodl-tmp/LIBERO-datasets/plans_libero_all_tasks.jsonl}"
PRETRAINED="${PRETRAINED:-/root/autodl-tmp/MoDE_Pretrained}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICES="${DEVICES:-1}"
# SHUTDOWN_MODE: always | on-success | never
SHUTDOWN_MODE="${SHUTDOWN_MODE:-always}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log}"

# Print the log path to the terminal, then send everything else to the file so
# the run is fully captured even after SSH disconnects.
echo "[train_and_shutdown] logging to: $LOG_FILE"
echo "[train_and_shutdown] watch with:  tail -f $LOG_FILE"
exec >>"$LOG_FILE" 2>&1

echo "=================================================================="
echo "=== train_and_shutdown.sh START $(date) ==="
echo "project=$PROJECT_DIR  env=$CONDA_ENV  devices=$DEVICES  batch_size=$BATCH_SIZE"
echo "plan_file=$PLAN_FILE"
echo "pretrained=$PRETRAINED  shutdown_mode=$SHUTDOWN_MODE"
echo "extra args: $*"
echo "=================================================================="

# ---- conda env (nohup starts a non-interactive shell -> init manually) ------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# ---- runtime env vars -------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR/LIBERO:$PROJECT_DIR"

cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }

# ---- train ------------------------------------------------------------------
python mode/training_libero.py \
  datamodule.plan_file="$PLAN_FILE" \
  model.start_from_pretrained=True \
  model.ckpt_path="$PRETRAINED" \
  devices="$DEVICES" batch_size="$BATCH_SIZE" \
  "$@"
CODE=$?

echo "=================================================================="
echo "=== training EXITED code=$CODE at $(date) ==="
echo "=================================================================="

# ---- shutdown ---------------------------------------------------------------
case "$SHUTDOWN_MODE" in
  always)
    echo "[shutdown] SHUTDOWN_MODE=always -> shutting down now"
    shutdown
    ;;
  on-success)
    if [ "$CODE" -eq 0 ]; then
      echo "[shutdown] training succeeded -> shutting down now"
      shutdown
    else
      echo "[shutdown] training failed (code=$CODE) -> NOT shutting down (kept for debugging)"
    fi
    ;;
  never)
    echo "[shutdown] SHUTDOWN_MODE=never -> leaving instance running"
    ;;
  *)
    echo "[shutdown] unknown SHUTDOWN_MODE='$SHUTDOWN_MODE' -> NOT shutting down"
    ;;
esac

exit "$CODE"
