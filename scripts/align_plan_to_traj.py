"""
最优方案 + z运动细化:
- 语义来自 plan(generate_atomic_plans, 正确);
- Grasp/Release 锚定到夹爪开合事件帧;
- 持物阶段(Grasp..Release 之间)的 Lift/Transport/Lower 用 EE z 高度曲线细化边界
  (上升=Lift, 高位平台=Transport, 下降=Lower); 其余原子线性插值。
画分子目标彩色轨迹。flip=True。
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
TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
ROOT = "/root/autodl-tmp/LIBERO-datasets"


def short(a):
    return a.replace("chefmate_8_", "").replace("flat_", "")


def smooth(x, k=5):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


# 1. plan
plan = None
for line in open(f"{ROOT}/plans_libero_all_tasks.jsonl"):
    o = json.loads(line)
    if TASK in o.get("task", ""):
        plan = o["plan"]["plan"]; break
atoms = [(s["action"], [short(x) for x in s.get("args", [])]) for s in plan]
N = len(atoms)

# 2. demo
h5 = [p for p in glob.glob(f"{ROOT}/{SUITE}/*.hdf5") if TASK in p][0]
d = h5py.File(h5, "r")["data"]["demo_0"]
g = np.array(d["actions"])[:, 6]
ee_pos = np.array(d["obs"]["ee_pos"], dtype=np.float64)
ee_z = ee_pos[:, 2]
imgs = np.array(d["obs"]["agentview_rgb"])
T = len(g)
closes = [t for t in range(1, T) if g[t-1] < 0 and g[t] > 0]
opens = [t for t in range(1, T) if g[t-1] > 0 and g[t] < 0]

# 3. 锚点 + 线性基线
grasp_idx = [i for i, a in enumerate(atoms) if a[0] == "Grasp"]
release_idx = [i for i, a in enumerate(atoms) if a[0] == "Release"]
anchors = {0: 0, N - 1: T - 1}
for gi, gt in zip(grasp_idx, closes):
    anchors[gi] = gt
for ri, ot in zip(release_idx, opens):
    anchors[ri] = ot
akeys = sorted(anchors)
atom_pos = np.zeros(N + 1)
for a0, a1 in zip(akeys, akeys[1:]):
    f0, f1 = anchors[a0], anchors[a1]
    for i in range(a0, a1 + 1):
        atom_pos[i] = f0 if a1 == a0 else f0 + (f1 - f0) * (i - a0) / (a1 - a0)
atom_pos[N] = T - 1


# 4. z 运动细化:对每个 Grasp->Release 的持物阶段,把 Lift/Transport/Lower 卡到 z 相位
def refine_pair(gi, ri):
    gf, rf = anchors[gi], anchors[ri]
    hold = list(range(gi + 1, ri))                 # 持物阶段的原子(Grasp与Release之间)
    acts = [atoms[i][0] for i in hold]
    L = rf - gf
    if L < 8 or not hold:
        return
    gw = max(4, int(0.12 * L))                     # 夹紧沉降窗口: Grasp 占 [gf, hs]
    hs = gf + gw                                    # 持物运动从此开始(Lift)
    seg = smooth(ee_z[hs:rf + 1])
    if len(seg) < 3:
        return
    zn = (seg - seg.min()) / (seg.max() - seg.min() + 1e-9)
    high = np.where(zn > 0.8)[0]
    if acts == ["Lift", "Transport", "Lower"] and len(high) >= 1:
        up_end = hs + int(high[0])
        down_start = min(hs + int(high[-1]), rf - 3)   # 保证 Lower>=3f
        up_end = min(max(up_end, hs + 1), down_start - 1)
        atom_pos[hold[0]] = hs            # Lift 起
        atom_pos[hold[1]] = up_end        # Transport 起
        atom_pos[hold[2]] = down_start    # Lower 起
        print(f"  [z细化] Grasp[{gf}-{hs}] Lift[{hs}-{up_end}] Transport[{up_end}-{down_start}] Lower[{down_start}-{rf}]")
    elif acts == ["Lift", "Lower"]:
        peak = hs + int(np.argmax(seg))
        atom_pos[hold[0]] = hs; atom_pos[hold[1]] = peak
        print(f"  [z细化] Grasp[{gf}-{hs}] Lift[{hs}-{peak}] Lower[{peak}-{rf}]")
    # 其余(如 Twist 单原子)不动


for gi, ri in zip(grasp_idx, release_idx):
    refine_pair(gi, ri)

ranges = [(int(round(atom_pos[i])), max(int(round(atom_pos[i])), int(round(atom_pos[i + 1])))) for i in range(N)]
print("=== 对齐+细化 结果 ===")
for i, ((act, args), (s, e)) in enumerate(zip(atoms, ranges)):
    print(f"  {i+1:2d}. {act}({','.join(args)})".ljust(38) + f"帧[{s}-{e}] ({e-s}f)")

# 5. 投影
bd = benchmark.get_benchmark_dict()[SUITE]()
n = getattr(bd, "n_tasks", None) or len(getattr(bd, "tasks", []))
idx = next((i for i in range(n) if TASK in str(getattr(bd.get_task(i), "bddl_file", ""))), 0)
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

# 6. 分子目标画
subgoals, cur = [], []
for i, (act, _) in enumerate(atoms):
    cur.append(i)
    if act == "Retreat":
        subgoals.append(cur); cur = []
if cur:
    subgoals.append(cur)
colors = cm.tab20(np.linspace(0, 1, 20))
fig, axes = plt.subplots(1, len(subgoals), figsize=(6.2 * len(subgoals), 6.2))
axes = np.atleast_1d(axes)
for sgi, sg in enumerate(subgoals):
    ax = axes[sgi]
    s0 = ranges[sg[0]][0]; e0 = ranges[sg[-1]][1]; mid = (s0 + e0) // 2
    ax.imshow(imgs[mid][::-1])
    ax.plot(c, r, "-", color="lightgray", lw=0.8, alpha=0.5)
    for i in sg:
        s, e = ranges[i]
        ax.plot(c[s:e + 1], r[s:e + 1], "-", color=colors[i % 20], lw=3.2, zorder=3,
                label=f"{i+1}. {atoms[i][0]}({','.join(atoms[i][1])}) [{s}-{e}]")
        ax.scatter([c[s]], [r[s]], s=60, color="white", edgecolors="black", zorder=5)
        ax.text(c[s], r[s], str(i + 1), fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=8, frameon=False)
    ax.set_title(f"Sub-goal {sgi+1} (frames {s0}-{e0})", fontsize=11)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.suptitle(f"PLAN-aligned + z-refined trajectory\n{TASK}", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/root/autodl-tmp/traj_plan_aligned_refined.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED", out)
