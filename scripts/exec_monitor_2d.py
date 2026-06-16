"""
执行时进度+健康度监控 demo(方法2:2D 轨迹跟踪误差)。
- 参照轨迹 = demo_0 的 EE 2D 投影 + plan对齐的逐帧原子动作标签(=大脑期望轨迹);
- 三种"实际执行":GOOD(小噪声跟随) / STALL(中途卡住不动) / DRIFT(抓取后跑偏);
- 在线单调最近邻跟踪 → 每步给出: 进度% + 当前原子动作 + 到参照轨迹的像素偏差;
- 自动诊断: OK / STALLED@step / OFF-TRACK@step。
说明: 实际执行用 demo_0 加扰动模拟(真实场景里就是策略 rollout 的 ee_pos)。
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
THR = 7.0  # 像素偏差阈值: 超过=跑偏


def short(a):
    return a.replace("chefmate_8_", "").replace("flat_", "")


def smooth(x, k=5):
    return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode="same")


# ---------- 1. plan + demo ----------
plan = None
for line in open(f"{ROOT}/plans_libero_all_tasks.jsonl"):
    o = json.loads(line)
    if TASK in o.get("task", ""):
        plan = o["plan"]["plan"]; break
atoms = [(s["action"], [short(x) for x in s.get("args", [])]) for s in plan]
N = len(atoms)

h5 = [p for p in glob.glob(f"{ROOT}/{SUITE}/*.hdf5") if TASK in p][0]
d = h5py.File(h5, "r")["data"]["demo_0"]
g = np.array(d["actions"])[:, 6]
ee = np.array(d["obs"]["ee_pos"], dtype=np.float64)
imgs = np.array(d["obs"]["agentview_rgb"])
T = len(g)
closes = [t for t in range(1, T) if g[t-1] < 0 and g[t] > 0]
opens = [t for t in range(1, T) if g[t-1] > 0 and g[t] < 0]

# ---------- 2. plan对齐 -> 逐帧原子动作标签 ----------
grasp_idx = [i for i, a in enumerate(atoms) if a[0] == "Grasp"]
release_idx = [i for i, a in enumerate(atoms) if a[0] == "Release"]
anchors = {0: 0, N - 1: T - 1}
for gi, gt in zip(grasp_idx, closes):
    anchors[gi] = gt
for ri, ot in zip(release_idx, opens):
    anchors[ri] = ot
akeys = sorted(anchors)
pos = np.zeros(N + 1)
for a0, a1 in zip(akeys, akeys[1:]):
    f0, f1 = anchors[a0], anchors[a1]
    for i in range(a0, a1 + 1):
        pos[i] = f0 if a1 == a0 else f0 + (f1 - f0) * (i - a0) / (a1 - a0)
pos[N] = T - 1
ranges = [(int(round(pos[i])), max(int(round(pos[i])), int(round(pos[i + 1])))) for i in range(N)]
step_label = [""] * T
for (act, args), (s, e) in zip(atoms, ranges):
    for t in range(s, min(e + 1, T)):
        step_label[t] = f"{act}({','.join(args)})"
for t in range(T):
    if not step_label[t]:
        step_label[t] = step_label[t - 1] if t else f"{atoms[0][0]}"

# ---------- 3. 投影矩阵 ----------
bd = benchmark.get_benchmark_dict()[SUITE]()
nt = getattr(bd, "n_tasks", None) or len(getattr(bd, "tasks", []))
idx = next((i for i in range(nt) if TASK in str(getattr(bd.get_task(i), "bddl_file", ""))), 0)
task = bd.get_task(idx)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=H, camera_widths=W)
env.reset()
sim = getattr(getattr(env, "env", env), "sim", None) or env.sim
wtp = camera_utils.get_camera_transform_matrix(sim, "agentview", H, W)
env.close()


def proj(ee3d):
    pix = camera_utils.project_points_from_world_to_camera(ee3d, wtp, H, W)
    r = (H - 1) - pix[:, 0].astype(float)
    c = pix[:, 1].astype(float)
    return np.stack([c, r], 1)   # (T,2) [x=col, y=row]


ref = proj(ee)                    # 参照 2D 轨迹

# ---------- 4. 造三种"实际执行"(3D 扰动后投影) ----------
rng = np.random.RandomState(0)
def make_good():
    return ee + rng.normal(0, 0.004, ee.shape)
def make_stall():
    a = ee.copy(); F = 168                       # 在搬运途中卡住
    a[F:] = ee[F] + rng.normal(0, 0.004, ee[F:].shape)
    return a
def make_drift():
    a = ee.copy(); F = 210                        # 抓到壶后方向跑偏
    k = np.arange(len(a) - F)[:, None]
    a[F:] = ee[F:] + np.array([0.0, 0.20, 0.05]) * (k / max(1, len(a) - F - 1))
    return a

scenarios = [("GOOD", make_good()), ("STALL", make_stall()), ("DRIFT", make_drift())]

# ---------- 5. 在线单调最近邻跟踪 ----------
def track(actual_pix, win=18):
    j = 0; prog = []; err = []; steps = []
    for t in range(len(actual_pix)):
        lo = j; hi = min(j + win, len(ref))
        dd = np.linalg.norm(ref[lo:hi] - actual_pix[t], axis=1)
        k = lo + int(np.argmin(dd))
        j = k
        prog.append(k / (len(ref) - 1)); err.append(float(dd.min())); steps.append(step_label[k])
    return np.array(prog), np.array(err), steps


def diagnose(prog, err, steps):
    if err.max() > THR:
        tb = int(np.argmax(err > THR))
        return f"OFF-TRACK @ '{steps[tb]}'  (t={tb}, {err[tb]:.1f}px)", "red"
    # 卡住: 进度长时间不前进且没到终点
    flat = 0
    for t in range(1, len(prog)):
        flat = flat + 1 if prog[t] - prog[t - 1] < 1e-3 else 0
        if flat > 25 and prog[t] < 0.9:
            return f"STALLED @ '{steps[t]}'  (progress {prog[t]*100:.0f}%)", "orange"
    return f"OK  (done, max err {err.max():.1f}px)", "green"


# ---------- 6. 画图 ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
for col, (name, actual_ee) in enumerate(scenarios):
    ap = proj(actual_ee)
    prog, err, steps = track(ap)
    verdict, vc = diagnose(prog, err, steps)

    # 上: 2D 空间叠加
    ax = axes[0, col]
    ax.imshow(imgs[T // 2][::-1])
    ax.plot(ref[:, 0], ref[:, 1], "--", color="white", lw=1.3, alpha=0.8, label="reference (brain plan)")
    ok = err <= THR
    ax.plot(ap[ok, 0], ap[ok, 1], ".", color="lime", ms=3, label="actual OK")
    if (~ok).any():
        ax.plot(ap[~ok, 0], ap[~ok, 1], ".", color="red", ms=4, label="actual OFF")
    ax.scatter([ap[0, 0]], [ap[0, 1]], c="cyan", s=40, edgecolors="k", zorder=5)
    ax.set_title(f"[{name}]  {verdict}", color=vc, fontsize=10, fontweight="bold")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.6)

    # 下: 偏差 + 进度 双轴, 背景按当前原子动作着色
    ax = axes[1, col]
    ax.plot(err, color="crimson", lw=1.6, label="tracking error (px)")
    ax.axhline(THR, ls="--", color="gray", lw=1, label=f"threshold {THR}px")
    ax.set_ylabel("error (px)", color="crimson"); ax.set_ylim(0, max(THR * 2, err.max() * 1.1 + 1))
    ax.set_xlabel("execution step t")
    ax2 = ax.twinx()
    ax2.plot(prog * 100, color="royalblue", lw=1.6, label="progress %")
    ax2.set_ylabel("progress %", color="royalblue"); ax2.set_ylim(0, 105)
    # 标注当前步(每隔取几个点)
    for tt in np.linspace(0, T - 1, 6).astype(int):
        ax.text(tt, ax.get_ylim()[1] * 0.92, steps[tt].split("(")[0], rotation=90,
                fontsize=6, ha="center", va="top", color="dimgray")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center left", fontsize=6, framealpha=0.6)

fig.suptitle("Execution monitor (Method 2: 2D trajectory tracking) — progress + health from comparing actual EE vs brain's planned path",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/root/autodl-tmp/exec_monitor.png"
fig.savefig(out, dpi=140)
print("=== 诊断结果 ===")
for (name, aee) in scenarios:
    ap = proj(aee); prog, err, steps = track(ap); v, _ = diagnose(prog, err, steps)
    print(f"  {name:6s} -> {v}")
print("SAVED", out)
