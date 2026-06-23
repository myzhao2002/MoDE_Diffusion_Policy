"""根因图: 同一 init0、同一掉落、物体停在同一位置。
立刻打断(REFLEX) -> 机械臂朝物体收敛、抓住; 暂停4秒(DELAY) -> 机械臂没去抓、距离一直在 0.1~0.3。
证明: 失败不是物体跑了(物体可达), 而是暂停后策略没回去重抓。数据来自 --trace 实测。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (t, dxy) 机械臂末端到目标物体的水平距离, 实测
reflex = [(105, .024), (120, .043), (135, .036), (150, .009), (165, .009)]
reflex_grasp = 171
delay = [(105, .025), (120, .088), (135, .101), (150, .128), (165, .171), (180, .205),
         (195, .206), (210, .205), (225, .165), (240, .157), (255, .177), (270, .188),
         (285, .147), (300, .204), (315, .322), (330, .321), (345, .318)]
delay_stall = [214]

fig, ax = plt.subplots(figsize=(8, 4.5))
rx, ry = zip(*reflex); dx, dy = zip(*delay)
ax.plot(rx, ry, "-o", color="#2e8b40", lw=2, label="REFLEX (interrupt, 0s pause) -> SUCCESS")
ax.plot(dx, dy, "-s", color="#c0392b", lw=2, label="DELAY (4s pause) -> FAIL")
ax.axvline(103, color="gray", ls="--", lw=1); ax.text(103, .34, "DROP", color="gray", ha="center", fontsize=9)
ax.axvline(reflex_grasp, color="#2e8b40", ls=":", lw=1.2)
ax.text(reflex_grasp, .30, "re-GRASP\n(success)", color="#2e8b40", ha="center", fontsize=8)
for s in delay_stall:
    ax.axvline(s, color="#c0392b", ls=":", lw=1.2)
    ax.text(s, .005, "STALL", color="#c0392b", ha="center", fontsize=8)
ax.axhline(0.04, color="black", ls="-.", lw=0.8, alpha=.5)
ax.text(330, .055, "graspable zone", fontsize=8, alpha=.7)
ax.set_xlabel("sim step  t"); ax.set_ylabel("arm-to-object horizontal distance (m)")
ax.set_title("Root cause: after a pause the policy fails to RE-APPROACH the object\n"
             "(object sits at the SAME reachable spot in both cases)", fontsize=10)
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=.25); ax.set_ylim(0, .37)
fig.tight_layout()
fig.savefig("/root/autodl-tmp/rootcause_distance.png", dpi=140)
print("SAVED /root/autodl-tmp/rootcause_distance.png")
