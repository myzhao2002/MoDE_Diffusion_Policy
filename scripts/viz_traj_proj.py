"""
3D EE 轨迹 → 2D agentview 图 投影可视化(训练侧监督信号的地基)。
用 robosuite 相机内外参把 demo 里夹爪的 3D 世界坐标投到 128x128 agentview 图上,
画出轨迹。图可能上下翻转,所以同时出 flip/no-flip 两版,看哪版轨迹贴着夹爪走。
"""
import os, glob
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import get_libero_path, benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils

H = W = 128
SUITE = "libero_10"
TASK_NAME = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"

# 1. demo 数据
h5path = [p for p in glob.glob(f"/root/autodl-tmp/LIBERO-datasets/{SUITE}/*.hdf5") if TASK_NAME in p][0]
print("hdf5:", h5path.split("/")[-1])
f = h5py.File(h5path, "r")
demo = f["data"]["demo_0"]
ee_pos = np.array(demo["obs"]["ee_pos"], dtype=np.float64)        # (T,3) 世界坐标
imgs = np.array(demo["obs"]["agentview_rgb"])                      # (T,128,128,3)
T = len(ee_pos)
print(f"T={T}  ee_pos min={ee_pos.min(0).round(3)} max={ee_pos.max(0).round(3)}")

# 2. 建对应任务的 env,拿 agentview 的 世界->像素 变换矩阵
bd = benchmark.get_benchmark_dict()[SUITE]()
n = getattr(bd, "n_tasks", None) or len(getattr(bd, "tasks", []))
idx = 0
for i in range(n):
    t = bd.get_task(i)
    if TASK_NAME in str(getattr(t, "name", "")) or TASK_NAME in str(getattr(t, "bddl_file", "")):
        idx = i
        break
task = bd.get_task(idx)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("env bddl:", bddl.split("/")[-1])
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=H, camera_widths=W)
env.reset()
sim = getattr(getattr(env, "env", env), "sim", None)
if sim is None:
    sim = env.sim
world_to_pix = camera_utils.get_camera_transform_matrix(sim, "agentview", H, W)
env.close()

# 3. 投影:世界坐标 -> 像素 [row, col]
pix = camera_utils.project_points_from_world_to_camera(ee_pos, world_to_pix, H, W)
print("pix row[min,max]=", round(float(pix[:, 0].min()), 1), round(float(pix[:, 0].max()), 1),
      " col[min,max]=", round(float(pix[:, 1].min()), 1), round(float(pix[:, 1].max()), 1))


def make_fig(flip):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    ts = np.linspace(0, T - 1, 6).astype(int)
    r = pix[:, 0].astype(float).copy()
    c = pix[:, 1].astype(float).copy()
    if flip:
        r = (H - 1) - r
    for ax, t in zip(axes.flat, ts):
        img = imgs[t][::-1] if flip else imgs[t]
        ax.imshow(img)
        ax.plot(c[:t + 1], r[:t + 1], "-", lw=1.6, color="cyan", alpha=0.9)   # 已走轨迹
        ax.scatter([c[t]], [r[t]], c="red", s=35, zorder=5)                    # 当前夹爪
        ax.set_title(f"t={t}", fontsize=8)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.suptitle(f"EE 3D->2D proj (flip={flip}) | {TASK_NAME[:42]}")
    fig.tight_layout()
    out = f"/root/autodl-tmp/traj_proj_flip{int(flip)}.png"
    fig.savefig(out, dpi=130)
    print("SAVED", out)


make_fig(False)
make_fig(True)
print("DONE")
