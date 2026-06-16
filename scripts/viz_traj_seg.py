"""
分阶段轨迹图(每段一个子图,全英文避免字体乱码)。
每个子图 = 该段中间帧做背景 + 全轨迹淡灰 + 当前段高亮(绿起点/红终点)。
flip=True (agentview 图上下翻转, 已验证)。
"""
import os, glob, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from libero.libero import get_libero_path, benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils

H = W = 128
SUITE = "libero_10"
TASK_NAME = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"


def short(name):
    # chefmate_8_frypan_1 -> frypan ; moka_pot_1 -> moka_pot
    s = name.replace("chefmate_8_", "").replace("flat_", "")
    s = "_".join([w for w in s.split("_") if not w.isdigit()])
    return s


# demo
h5path = [p for p in glob.glob(f"/root/autodl-tmp/LIBERO-datasets/{SUITE}/*.hdf5") if TASK_NAME in p][0]
f = h5py.File(h5path, "r"); demo = f["data"]["demo_0"]
ee_pos = np.array(demo["obs"]["ee_pos"], dtype=np.float64)
imgs = np.array(demo["obs"]["agentview_rgb"])
T = len(ee_pos)

# segments
segj = [p for p in glob.glob(f"/root/autodl-tmp/LIBERO-datasets/segments_v3/{SUITE}/*.json")
        if TASK_NAME in p and "checkpoint" not in p][0]
segs = json.load(open(segj))["demos"]["demo_0"]["segments"]
print(f"{len(segs)} segments")

# projection
bd = benchmark.get_benchmark_dict()[SUITE]()
n = getattr(bd, "n_tasks", None) or len(getattr(bd, "tasks", []))
idx = next((i for i in range(n) if TASK_NAME in str(getattr(bd.get_task(i), "bddl_file", ""))), 0)
task = bd.get_task(idx)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=H, camera_widths=W)
env.reset()
sim = getattr(getattr(env, "env", env), "sim", None) or env.sim
wtp = camera_utils.get_camera_transform_matrix(sim, "agentview", H, W)
env.close()
pix = camera_utils.project_points_from_world_to_camera(ee_pos, wtp, H, W)
r = (H - 1) - pix[:, 0].astype(float)
c = pix[:, 1].astype(float)

# per-segment grid
ncols = 4
nrows = int(np.ceil(len(segs) / ncols))
colors = cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
axes = np.array(axes).reshape(-1)
for i, seg in enumerate(segs):
    ax = axes[i]
    s = int(seg["start"]); e = int(min(seg["end"], T - 1))
    mid = (s + e) // 2
    ax.imshow(imgs[mid][::-1])                                  # frame during this segment
    ax.plot(c, r, "-", color="lightgray", lw=0.8, alpha=0.6)   # full traj faint
    ax.plot(c[s:e + 1], r[s:e + 1], "-", color=colors[i % 10], lw=3, zorder=3)  # this segment
    ax.scatter([c[s]], [r[s]], color="lime", s=55, edgecolors="black", zorder=5)   # start
    ax.scatter([c[e]], [r[e]], color="red", s=55, edgecolors="black", zorder=5)    # end
    args = ",".join(short(a) for a in (seg.get("args", []) or []))
    ax.set_title(f"{i+1}. {seg['action']}({args})\nframes [{s}-{e}]  ({e-s+1}f)", fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
for j in range(len(segs), len(axes)):
    axes[j].axis("off")
fig.suptitle(f"Per-segment EE trajectory | {TASK_NAME}\n"
             f"green=seg start, red=seg end, gray=full path", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/root/autodl-tmp/traj_seg_panels.png"
fig.savefig(out, dpi=140)
print("SAVED", out)
print("DONE")
