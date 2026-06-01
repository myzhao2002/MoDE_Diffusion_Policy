"""
One-off diagnostic: is the task_plan_fuser in the trained checkpoint actually
trained, and is it being restored at eval time?

Run on autodl:
    conda activate lerobot
    python inspect_fuser.py
"""
import torch

CKPT = "/root/autodl-tmp/MoDE_ckpts/best-epoch=18-val_act/lang_act_loss_pp=0.0207.ckpt"

print(f"Loading checkpoint: {CKPT}")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)

# Lightning checkpoints store params under "state_dict"; fall back if not.
sd = ck.get("state_dict", ck)

fuser_keys = [k for k in sd if "task_plan_fuser" in k]
print(f"\n# task_plan_fuser keys in checkpoint state_dict: {len(fuser_keys)}")
for k in fuser_keys:
    w = sd[k].float()
    print(f"  {k:50s} shape={tuple(w.shape)} mean={w.mean():.6f} std={w.std():.6f}")

# Show the first few raw values of the first Linear weight so we can compare
# against the eval-loaded model and a fresh init.
wkey = "task_plan_fuser.net.0.weight"
if wkey in sd:
    flat = sd[wkey].float().flatten()[:8]
    print(f"\n# {wkey} first 8 values (CHECKPOINT):")
    print("  ", flat.tolist())
else:
    print(f"\n# {wkey} NOT FOUND in checkpoint state_dict!")
    print("  -> the fuser was never saved; eval will use a fresh random init -> guaranteed train/eval mismatch")

# Compare against a fresh init of the same module for reference.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mode.models.networks.task_plan_fuser import TaskPlanFuser
    torch.manual_seed(0)
    fresh = TaskPlanFuser()
    fw = fresh.net[0].weight.detach().float()
    print(f"\n# fresh TaskPlanFuser().net[0].weight (seed=0): mean={fw.mean():.6f} std={fw.std():.6f}")
    print("  first 8 values:", fw.flatten()[:8].tolist())
except Exception as e:
    print(f"\n(could not import TaskPlanFuser for reference: {e})")

# Also report EMA weights if present (the 'best' ckpt may use EMA params).
cb = ck.get("callbacks", {})
ema = None
for key, val in cb.items():
    if "EMA" in str(key) and isinstance(val, dict) and "ema_weights" in val:
        ema = val["ema_weights"]
        break
if ema is not None:
    print(f"\n# EMA weights present in checkpoint: {len(ema)} tensors (list form).")
    print("  NOTE: eval load (use_ema_weights=False) uses the NON-ema state_dict above.")
else:
    print("\n# No EMA weights found in checkpoint callbacks.")
