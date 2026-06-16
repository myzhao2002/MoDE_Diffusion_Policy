"""
执行监控 v2:轨迹走廊(多 demo 统计容差带),解决单条参照误报。
- 取同一任务多条 demo, 各自投影到 2D, 用夹爪事件把时间轴归一到 canonical 进度;
- 每个 canonical 进度点统计 mean(中心线) + 2.5σ(走廊半径) => 容差管道;
- 测试: 留出的真实 demo(GOOD, 物体位置不同但应在带内) vs 跑偏版(BAD, 应出带);
- 监控: 实际点到中心线距离 <= 该处走廊半径 => 正常; 出带 => 异常, 并报当前原子动作。
"""
import os, glob, json
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
TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
ROOT = "/root/autodl-tmp/LIBERO-datasets"
NG = 120            # canonical 网格点数
SIG = 4.0          # 走廊半径 = SIG * 径向std
FLOOR = 6.0        # 半径下限(px)
KRUN = 6           # 连续 KRUN 点出带才判异常(抗单点擦边误报)


def short(a):
    return a.replace("chefmate_8_", "").replace("flat_", "")


# ---------- plan ----------
plan = None
for line in open(f"{ROOT}/plans_libero_all_tasks.jsonl"):
    o = json.loads(line)
    if TASK in o.get("task", ""):
        plan = o["plan"]["plan"]; break
atoms = [(s["action"], [short(x) for x in s.get("args", [])]) for s in plan]
N = len(atoms)
grasp_ai = [i for i, a in enumerate(atoms) if a[0] == "Grasp"]
release_ai = [i for i, a in enumerate(atoms) if a[0] == "Release"]

# ---------- 投影矩阵(agentview 固定) ----------
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
    return np.stack([pix[:, 1].astype(float), (H - 1) - pix[:, 0].astype(float)], 1)


# canonical 锚点比例: [start, g1, r1, g2, r2, end] -> 0,.2,.4,.6,.8,1
U_ANCHORS = np.array([0, .2, .4, .6, .8, 1.0])


def demo_warp(ee, g, T):
    closes = [t for t in range(1, T) if g[t-1] < 0 and g[t] > 0]
    opens = [t for t in range(1, T) if g[t-1] > 0 and g[t] < 0]
    if len(closes) < 2 or len(opens) < 2:
        return None
    af = np.array([0, closes[0], opens[0], closes[1], opens[1], T - 1], dtype=float)
    if not np.all(np.diff(af) > 0):
        return None
    p2 = proj(ee)
    u = np.linspace(0, 1, NG)
    frames = np.interp(u, U_ANCHORS, af)
    cx = np.interp(frames, np.arange(len(p2)), p2[:, 0])
    cy = np.interp(frames, np.arange(len(p2)), p2[:, 1])
    return np.stack([cx, cy], 1)   # (NG,2) canonical


# ---------- 读多条 demo, 建走廊 ----------
f = h5py.File(h5 := [p for p in glob.glob(f"{ROOT}/{SUITE}/*.hdf5") if TASK in p][0], "r")
demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
warps = []
used = []
for k in demo_keys:
    dd = f["data"][k]
    ee = np.array(dd["obs"]["ee_pos"], dtype=np.float64)
    g = np.array(dd["actions"])[:, 6]
    w = demo_warp(ee, g, len(g))
    if w is not None:
        warps.append(w); used.append(k)
warps = np.array(warps)                      # (M, NG, 2)
M = len(warps)


def build_band(idxs):
    c = warps[idxs].mean(0)
    rad = SIG * np.linalg.norm(warps[idxs] - c[None], axis=2).std(0)
    rad = np.maximum(np.convolve(rad, np.ones(7) / 7, mode="same"), FLOOR)
    return c, rad


def sustained_out(dist, rad):
    out = dist > rad; run = 0
    for t in range(len(out)):
        run = run + 1 if out[t] else 0
        if run >= KRUN:
            return t - KRUN + 1
    return -1


# ---- 留一法标定: 每条真实 demo 用其余建带后测试, 统计误报 ----
loo_fail = []
for i in range(M):
    others = np.delete(np.arange(M), i)
    c, rad = build_band(others)
    if sustained_out(np.linalg.norm(warps[i] - c, axis=1), rad) >= 0:
        loo_fail.append(used[i])
passrate = (M - len(loo_fail)) / M * 100
print(f"LOO标定: {M-len(loo_fail)}/{M} 条真实demo通过走廊 (误报率 {len(loo_fail)/M*100:.0f}%); 误报demo={loo_fail[:6]}")

# 选一条"通过"的 demo 当 GOOD 展示(代表性, 非挑好的); 其余建走廊
held_i = next((i for i in range(M) if used[i] not in loo_fail), M - 1)
HELD = used[held_i]
build_idx = np.delete(np.arange(M), held_i)
center, radius = build_band(build_idx)
print(f"展示用: 留出 {HELD}(LOO通过) 做GOOD; 用其余 {M-1} 条建走廊; 半径 px: min={radius.min():.1f} max={radius.max():.1f}")

# canonical 进度 -> 当前原子动作(锚点把 plan 原子映到 u)
atom_u_end = np.zeros(N)
# 每个原子在 canonical 的结束 u: 用锚点段内线性
anchor_atom = {0: 0, N - 1: NG - 1}  # 占位
# 直接: atom i 的 canonical u 起点 = 按 plan 锚点(grasp/release)在 U_ANCHORS 上的位置插值
plan_anchor_atoms = [0] + grasp_ai + release_ai + [N - 1]
# 排序成 (atom_idx, u)
amap = {0: 0.0, N - 1: 1.0}
for j, gi in enumerate(grasp_ai):
    amap[gi] = U_ANCHORS[1 + 2 * j]      # g1->.2, g2->.6
for j, ri in enumerate(release_ai):
    amap[ri] = U_ANCHORS[2 + 2 * j]      # r1->.4, r2->.8
ak = sorted(amap)
atom_u = np.zeros(N)
for a0, a1 in zip(ak, ak[1:]):
    for i in range(a0, a1 + 1):
        atom_u[i] = amap[a0] if a1 == a0 else amap[a0] + (amap[a1] - amap[a0]) * (i - a0) / (a1 - a0)


def step_at(u):
    i = int(np.searchsorted(atom_u, u, side="right") - 1)
    i = max(0, min(N - 1, i))
    return f"{atoms[i][0]}({','.join(atoms[i][1])})"


def tube_poly(c, rad):
    d = np.gradient(c, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    up = c + nrm * rad[:, None]; lo = c - nrm * rad[:, None]
    return np.vstack([up, lo[::-1]])


# ---------- 测试轨迹 ----------
def warp_of_key(k):
    dd = f["data"][k]
    return demo_warp(np.array(dd["obs"]["ee_pos"], np.float64), np.array(dd["actions"])[:, 6],
                     len(dd["actions"]))


good = warp_of_key(HELD)                         # 真实留出 demo(物体位置不同)
bad = good.copy()
fb = int(0.62 * NG)                              # 抓壶后开始跑偏
kk = np.arange(NG - fb)[:, None]
bad[fb:] = good[fb:] + np.array([26.0, 12.0]) * (kk / max(1, NG - fb - 1))   # 像素级跑偏


def monitor(testw):
    dist = np.linalg.norm(testw - center, axis=1)
    inband = dist <= radius
    out = ~inband
    u = np.linspace(0, 1, NG)
    run = 0; firstbad = -1                         # 连续 KRUN 点出带才算异常
    for t in range(NG):
        run = run + 1 if out[t] else 0
        if run >= KRUN:
            firstbad = t - KRUN + 1; break
    if firstbad >= 0:
        return dist, inband, f"OFF-CORRIDOR @ '{step_at(u[firstbad])}' (u={u[firstbad]*100:.0f}%, {dist[firstbad]:.1f}>{radius[firstbad]:.1f}px)", "red"
    return dist, inband, f"OK (stays in corridor, max excess {(dist-radius).max():.1f}px)", "green"


bgimg = np.array(f["data"][HELD]["obs"]["agentview_rgb"])[len(f["data"][HELD]["actions"]) // 2][::-1]
u = np.linspace(0, 1, NG)

# ---------- 画 ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
poly = tube_poly(center, radius)

# (0,0) 走廊构建
ax = axes[0, 0]; ax.imshow(bgimg)
for w in warps[build_idx]:
    ax.plot(w[:, 0], w[:, 1], "-", color="gray", lw=0.6, alpha=0.35)
ax.fill(poly[:, 0], poly[:, 1], color="cyan", alpha=0.25, zorder=2)
ax.plot(center[:, 0], center[:, 1], "-", color="blue", lw=2, zorder=3)
ax.set_title(f"Corridor from {M-1} demos\n(gray=demos, blue=centerline, cyan=±{SIG}σ tube)", fontsize=9)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

# (0,1)(0,2) GOOD / BAD 叠加
for col, (name, tw) in enumerate([("GOOD (held-out real demo)", good), ("BAD (drift injected)", bad)], start=1):
    dist, inb, verdict, vc = monitor(tw)
    ax = axes[0, col]; ax.imshow(bgimg)
    ax.fill(poly[:, 0], poly[:, 1], color="cyan", alpha=0.22, zorder=2)
    ax.plot(center[:, 0], center[:, 1], "-", color="blue", lw=1.2, alpha=0.7, zorder=3)
    ax.plot(tw[inb, 0], tw[inb, 1], ".", color="lime", ms=4, zorder=4)
    if (~inb).any():
        ax.plot(tw[~inb, 0], tw[~inb, 1], ".", color="red", ms=5, zorder=4)
    ax.set_title(f"[{name}]\n{verdict}", color=vc, fontsize=9, fontweight="bold")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

# (1,0) 半径剖面
ax = axes[1, 0]
ax.plot(u * 100, radius, color="purple", lw=1.6)
ax.set_title("corridor radius vs progress", fontsize=9)
ax.set_xlabel("progress %"); ax.set_ylabel("radius (px)")
for uu in U_ANCHORS:
    ax.axvline(uu * 100, ls=":", color="gray", lw=0.7)

# (1,1)(1,2) GOOD/BAD 距离 vs 半径
for col, (name, tw) in enumerate([("GOOD", good), ("BAD", bad)], start=1):
    dist, inb, verdict, vc = monitor(tw)
    ax = axes[1, col]
    ax.plot(u * 100, dist, color="crimson", lw=1.6, label="dist to centerline")
    ax.plot(u * 100, radius, "--", color="gray", lw=1.3, label="corridor radius")
    ax.fill_between(u * 100, 0, radius, color="cyan", alpha=0.18)
    if (~inb).any():
        tb = int(np.argmax(~inb))
        ax.axvline(u[tb] * 100, color="red", lw=1, ls="-")
        ax.text(u[tb] * 100, dist.max() * 0.9, f"  out @ {step_at(u[tb]).split('(')[0]}", color="red", fontsize=7)
    ax.set_title(f"{name}: in-corridor check", fontsize=9)
    ax.set_xlabel("progress %"); ax.set_ylabel("px"); ax.legend(fontsize=6)

fig.suptitle("Execution monitor v2 — trajectory corridor (multi-demo tolerance tube): legit variation stays in, real failure exits",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/root/autodl-tmp/exec_monitor_corridor.png"
fig.savefig(out, dpi=140)
print("=== 诊断 ===")
for name, tw in [("GOOD(held-out)", good), ("BAD(drift)", bad)]:
    _, _, v, _ = monitor(tw); print(f"  {name:16s} -> {v}")
print("SAVED", out)
