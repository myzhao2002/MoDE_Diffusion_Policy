"""
注意力可视化(实验四 v2):跑 1 条 LIBERO episode,捕获视觉查询分步规划交叉注意力的
权重,画"时间 × 规划步"热力图,看 xattn 是否学成"进度指针"。

复用 mode_evaluate_libero 的 env/model 搭建,monkey-patch model.step 抓 attn。
依赖 lang_perceiver.py 里 VisualQueryCrossAttention 的埋点 self.attn_weights。

用法(CLI 同 eval):
  python scripts/viz_attn_libero.py train_folder=... checkpoint=... \
    plan_file=... log_wandb=False viz_task=0 viz_bench=libero_10
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from mode.evaluation.utils import get_default_mode_and_env
from mode.evaluation.mode_evaluate_libero import EvaluateLibero, get_log_dir


@hydra.main(config_path="../conf", config_name="mode_evaluate_libero")
def main(cfg):
    seed_everything(0, workers=True)
    bench = cfg.get("viz_bench", "libero_10")
    task_idx = int(cfg.get("viz_task", 0))
    out_png = cfg.get("viz_out", "/root/autodl-tmp/attn_viz.png")

    print("[viz] train_folder =", cfg.train_folder)
    print("[viz] checkpoint   =", cfg.checkpoint)
    model, _, dm, _ = get_default_mode_and_env(
        train_folder=cfg.train_folder, dataset_path=cfg.dataset_path,
        checkpoint=cfg.checkpoint, env=42, lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
        prep_dm_and_deps=False)
    model = model.to(cfg.device)
    model.eval()
    transforms = hydra.utils.instantiate(dm.transforms)

    ev = EvaluateLibero(
        model=model, transforms=transforms, log_dir=get_log_dir(cfg.log_dir),
        benchmark_name=bench, num_sequences=cfg.num_sequences, num_videos=0,
        max_steps=cfg.max_steps, n_eval=1,
        task_embedding_format=cfg.task_embedding_format, device=cfg.device,
        plan_file=cfg.get("plan_file", ""))

    # ---- monkey-patch model.step:每步抓 static 相机第0层交叉注意力权重 ----
    attn_log = []
    orig_step = model.step

    def patched_step(data, goal):
        out = orig_step(data, goal)
        try:
            aw = model.static_xattn.layers[0][0].attn_weights  # [B, heads, Q, S]
            attn_log.append(aw[0].mean(dim=(0, 1)).float().cpu().numpy())  # 头+query 平均 -> [S]
        except Exception as e:
            print("[viz] capture failed:", e)
        return out
    model.step = patched_step

    # ---- 取 1 个任务 + 它的 plan(做 y 轴标签)----
    task_i = ev.benchmark_instance.get_task(task_idx)
    task_emb = ev.benchmark_instance.task_embs[task_idx]
    demo_name = os.path.splitext(os.path.basename(
        ev.benchmark_instance.get_task_demonstration(task_idx)))[0]
    plan_text = ev.plan_by_task.get(demo_name, task_i.language)
    steps = [s.strip() for s in str(plan_text).split("->") if s.strip()]
    print(f"[viz] task = {task_i.language}")
    print(f"[viz] plan = {plan_text}")

    # ---- 跑 1 条 episode(n_eval=1)----
    ev.evaluate_task(model, task_i, task_emb, f"viz_p{task_idx}", task_idx, store_video=0)

    if not attn_log:
        print("[viz] !! 没抓到任何注意力,检查埋点")
        return
    A = np.stack(attn_log)  # [T, S]
    np.save(out_png.replace(".png", ".npy"), A)
    S = A.shape[1]
    labels = (steps + [f"pad{i}" for i in range(S)])[:S]

    plt.figure(figsize=(max(10, A.shape[0] * 0.06), 4))
    plt.imshow(A.T, aspect="auto", cmap="viridis", origin="upper")
    plt.yticks(range(S), labels, fontsize=7)
    plt.xlabel("rollout step (每步=1次推理≈10环境步)")
    plt.ylabel("规划步骤")
    plt.colorbar(label="注意力权重")
    plt.title(f"xattn over plan steps | {str(task_i.language)[:60]}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"[viz] SAVED {out_png}  shape={A.shape}")
    # 文本版概览:每步被关注最多的规划步索引
    argmax_seq = A.argmax(axis=1)
    print("[viz] 每个rollout步关注的plan步索引序列(看是否随时间递增=进度指针):")
    print(argmax_seq.tolist())


if __name__ == "__main__":
    main()
