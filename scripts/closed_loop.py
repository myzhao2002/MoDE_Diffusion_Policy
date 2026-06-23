"""
真闭环 第1块: 在线监控(实时触发器)。
执行中逐步判物体状态, 一旦"掉落/抓空(DROP)"或"卡住(STALL)"当场报警 ——
这是触发大脑重规划的信号。(大脑重规划 + 小脑重执行 是下一块)

在线事件:
  GRASP_OK(o): 物体 o 变成"被握住"(近 EE 且抬离静止高度)
  PLACED(o):   o 从握住->放下 且 在目标附近 = 成功放置
  DROP(o):     o 从握住->放下 但 不在目标附近 = 抓空/滑脱 失败  <-- 触发
  STALL:       超过 stall_k 步没有任何进展 = 卡住            <-- 触发

CLI(先验证在线检测在注入失败那一刻实时报):
  python scripts/closed_loop.py --train_folder ... --checkpoint ... \
    --inject open_gripper --inject_step 95 --inject_len 25 --max_steps 500
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import subprocess
import time
import numpy as np
import torch
import cv2
from libero.libero.envs import OffScreenRenderEnv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.inject_and_monitor import load_models, PLANS
from scripts.build_progress_labels import match_plan
from scripts.qwen_brain import QwenBrain


class OnlineMonitor:
    """逐步在线判: 物体被握住/放置/掉落 + 卡住。失败当场返回事件。"""
    def __init__(self, objs, rest_z, targets, near=0.06, lift=0.02, stall_k=110, smart_stall=False):
        self.objs = objs
        self.rest_z = rest_z          # o -> 静止高度
        self.targets = targets        # o -> 目标物体名(basket)
        self.held = {o: False for o in objs}
        self.placed = {o: False for o in objs}
        self.last_event_t = 0
        self.stalled = False          # 锁存: STALL 只报一次, 等真事件再解锁
        self.near, self.lift, self.stall_k = near, lift, stall_k
        # 进度感知: 机械臂朝未放好物体靠近(说明在自愈)就别误判卡死
        self.smart_stall = smart_stall
        self.best_dist = float("inf")  # 自上次事件以来, 机械臂到未放好物体的最小距离
        self.prog_eps = 0.03           # 距离再缩小 >3cm 算"有进展", 重置卡死计时

    def update(self, t, ee, objpos):
        evs = []
        real = False
        for o in self.objs:
            p = objpos[o]
            held_now = (np.linalg.norm(ee[:2] - p[:2]) < self.near) and (p[2] > self.rest_z[o] + self.lift)
            tgt = objpos.get(self.targets.get(o))
            at_tgt = tgt is not None and np.linalg.norm(p[:2] - tgt[:2]) < 0.09
            if held_now and not self.held[o]:
                evs.append(("GRASP_OK", o)); self.last_event_t = t; real = True
            elif self.held[o] and not held_now:
                if at_tgt:
                    evs.append(("PLACED", o)); self.placed[o] = True; self.last_event_t = t; real = True
                else:
                    evs.append(("DROP", o)); self.last_event_t = t; real = True
            self.held[o] = held_now
        if real:
            self.stalled = False
            self.best_dist = float("inf")   # 新事件 -> 重新计进度
        # 进度感知: 机械臂正接近某个未放好物体(距离创新低) = 在自愈, 算活动, 别判卡死
        if self.smart_stall and not real:
            unplaced = [o for o in self.objs if not self.placed[o] and o in objpos]
            if unplaced:
                dmin = min(np.linalg.norm(ee[:2] - objpos[o][:2]) for o in unplaced)
                if dmin < self.best_dist - self.prog_eps:
                    self.best_dist = dmin
                    self.last_event_t = t        # 有靠近进展 -> 重置卡死计时
        if not real and not self.stalled and (t - self.last_event_t > self.stall_k):
            evs.append(("STALL", None)); self.stalled = True
        return evs


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _msg(kind, o):
    return {
        "GRASP_OK": (f"grasped {o}", (60, 180, 60)),
        "PLACED":   (f"OK placed {o}", (60, 180, 60)),
        "DROP":     (f"!! DROP {o}  -> detect", (40, 40, 230)),
        "STALL":    ("!! STALL  -> detect", (40, 40, 230)),
        "PENDING":  (f"detected {o} - finish current chunk first", (40, 120, 230)),
        "CHUNK_DONE": ("chunk done -> stop cerebellum, call brain", (0, 165, 255)),
        "THINKING": (f"BRAIN (Qwen-VL) replanning {o or ''}", (0, 165, 255)),
        "RECOVER":  (f">> REPLAN: redo {o}", (0, 140, 255)),
    }.get(kind, (kind, (200, 200, 200)))


def compose_video(frames, events, inject_win, success, out_path, ffmpeg=None, think_spans=None):
    T = len(frames); vh = vw = 224; pw = 320; bh = 46; H, W = bh + vh, vw + pw
    evs = sorted(events)
    cur, ei = ("executing ...", (190, 190, 190)), 0
    placed = []
    tmp = out_path.replace(".mp4", "_raw.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
    log_lines = []
    for t, f in enumerate(frames):
        while ei < len(evs) and evs[ei][0] <= t:
            _, k, o = evs[ei]
            cur = _msg(k, o)
            log_lines.append((evs[ei][0], k, o))
            if k == "PLACED":
                placed.append(o)
            ei += 1
        cvb = np.full((H, W, 3), 28, np.uint8)
        cvb[bh:bh + vh, :vw] = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        in_think = any(s <= t < e for s, e in (think_spans or []))
        inj = (inject_win[0] <= t < inject_win[1]) and not in_think
        cv2.putText(cvb, cur[0], (6, 30), FONT, 0.6, cur[1], 2)
        cv2.putText(cvb, f"t={t}/{T}" + ("  [INJECT]" if inj else ""), (W - 150, 18), FONT, 0.42,
                    (40, 40, 230) if inj else (170, 170, 170), 1)
        if in_think:  # 大脑思考期间小脑保持不动(真实仿真, 物体按物理下落)
            cv2.putText(cvb, "arm HOLDING (brain replanning)", (16, bh + vh - 12), FONT, 0.45,
                        (0, 165, 255), 1)
        # 右侧事件日志(最近若干)
        cv2.putText(cvb, "events:", (vw + 8, bh + 12), FONT, 0.4, (170, 170, 170), 1)
        for j, (et, k, o) in enumerate(log_lines[-13:]):
            col = _msg(k, o)[1]
            cv2.putText(cvb, f"t{et}: {k} {o or ''}", (vw + 8, bh + 30 + j * 14), FONT, 0.36, col, 1)
        wr.write(cvb)
    # 末尾定格 30 帧显示结果
    last = cvb.copy()
    res = "SUCCESS (task done)" if success else "FAILURE"
    cv2.rectangle(last, (0, 0), (W, bh), (40, 120, 40) if success else (40, 40, 160), -1)
    cv2.putText(last, res, (6, 32), FONT, 0.7, (255, 255, 255), 2)
    for _ in range(30):
        wr.write(last)
    wr.release()
    if ffmpeg:
        try:
            subprocess.run([ffmpeg, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
                           check=True, capture_output=True, timeout=120)
            os.remove(tmp); return out_path
        except Exception as e:
            print("[closed] ffmpeg convert failed:", e)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_folder", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup")
    ap.add_argument("--inject", default="open_gripper")
    ap.add_argument("--inject_step", type=int, default=95)
    ap.add_argument("--inject_len", type=int, default=25)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--recover", action="store_true", help="检测到失败时换子计划重做(测恢复机制)")
    ap.add_argument("--max_recover", type=int, default=4)
    ap.add_argument("--brain", default="heuristic", choices=["heuristic", "qwen"], help="决策大脑: 启发式 or 千问VL看图决策")
    ap.add_argument("--interrupt", action="store_true",
                    help="检测到失败立刻打断重规划(默认: 先把当前动作块剩余步执行完再到块边界重规划)")
    ap.add_argument("--brain_delay_s", type=float, default=0.0,
                    help="人为设定的大脑时延(秒): 恢复时机械臂空等这么久(用于扫时延曲线, 与真大脑解耦)")
    ap.add_argument("--trace", action="store_true", help="每15步打印目标物体/机械臂/距离, 诊断暂停为何抓不到")
    ap.add_argument("--smart_stall", action="store_true",
                    help="进度感知卡死检测: 机械臂正朝未放好物体靠近(在自愈)就不判卡死, 避免打断自愈策略")
    ap.add_argument("--dump_tag", default="", help="非空则把恢复后小脑即将看到的obs图(agentview+腕部)存为png对比")
    ap.add_argument("--freeze_arm", action="store_true",
                    help="暂停期间把机械臂关节强行锁死(完全不动), 只让物体随物理下落; 验证'手臂不动则暂停无害'")
    args = ap.parse_args()

    model, ev = load_models(args.train_folder, args.checkpoint, args.device)
    brain = QwenBrain(device=f"cuda:{args.device}") if args.brain == "qwen" else None
    bd = ev.benchmark_instance
    idx = next((i for i in range(bd.get_num_tasks()) if args.task in str(bd.get_task(i).bddl_file)), 0)
    task_i = bd.get_task(idx); task_emb = bd.task_embs[idx]
    demo_name = os.path.splitext(os.path.basename(bd.get_task_demonstration(idx)))[0]
    plan_text = ev.plan_by_task.get(demo_name, task_i.language)
    atoms = match_plan(PLANS, demo_name) or []
    bddl = os.path.join(ev.bddl_folder, task_i.problem_folder, task_i.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
    body_names = list(env.sim.model.body_names)
    objs = sorted({a for _, ar in atoms for a in ar if a not in ("grasp_pose", "twist_pose", "clockwise", "counterclockwise")})
    bid = {o: env.sim.model.body_name2id(b) for o in objs
           for b in [next((x for x in body_names if o.lower() in x.lower()), None)] if b}
    # 目标: 每个被抓物体 -> 它 Transport/Lower 的目标(basket)
    targets = {}
    for a, ar in atoms:
        if a in ("Transport", "Lower") and len(ar) > 1 and ar[0]:
            targets.setdefault(ar[0], ar[1])
    grasp_objs = [ar[0] for a, ar in atoms if a == "Grasp" and ar[0] in bid]
    # 每个物体的子计划文本(它涉及的原子, 用于恢复时让小脑重做这个物体)
    def atxt(a, ar):
        return f"{a}({', '.join(ar)})" if ar else f"{a}()"
    subplans = {o: " -> ".join(atxt(a, ar) for a, ar in atoms if o in ar) for o in grasp_objs}

    init_states = torch.load(os.path.join(ev.init_states_folder, task_i.problem_folder, task_i.init_states_file))
    env.reset(); model.reset()
    obs = env.set_init_state(init_states[args.init % len(init_states)])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    rest_z = {o: float(env.sim.data.body_xpos[bid[o]][2]) for o in bid}
    mon = OnlineMonitor(grasp_objs, rest_z, targets, smart_stall=args.smart_stall)
    print(f"[closed] task={task_i.language}")
    print(f"[closed] grasp objects={grasp_objs}, targets={targets}")
    print(f"[closed] inject={args.inject}@{args.inject_step}(+{args.inject_len})")

    cur_plan = plan_text; n_recover = 0
    done = False
    frames, events, think_spans = [], [], []
    inj_win = (args.inject_step, args.inject_step + args.inject_len) if args.inject != "none" else (-1, -1)
    FPS = 20.0
    pending = None     # 待处理失败(kind,o): 检测到后先把当前动作块剩余步执行完, 到块边界再重规划
    for t in range(args.max_steps):
        data, goal = ev.process_env_obs(obs, task_emb, task_i.language, cur_plan)
        action = model.step(data, goal).cpu().numpy().reshape(-1)[:7]
        if args.inject != "none" and args.inject_step <= t < args.inject_step + args.inject_len:
            if args.inject == "open_gripper":
                action[6] = -1.0
            elif args.inject == "action_noise":
                action[:6] += np.random.randn(6) * 0.5
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1].copy())
        fi = len(frames) - 1            # 当前帧在视频里的索引(用于事件对齐)
        objpos = {o: env.sim.data.body_xpos[b].copy() for o, b in bid.items()}
        ee = np.array(obs.get("robot0_eef_pos", [0, 0, 0]))
        # —— 轨迹追踪(诊断暂停为何抓不到): 每15步打印 目标物体/机械臂/距离 ——
        if args.trace and t % 15 == 0:
            tg = next((x for x in grasp_objs if not mon.placed[x]), None)
            if tg in objpos:
                op = objpos[tg]; dxy = float(np.linalg.norm(ee[:2] - op[:2]))
                print(f"[TRACE] t={t:3d} obj={tg} op=({op[0]:.3f},{op[1]:.3f},{op[2]:.3f}) "
                      f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) dxy={dxy:.3f} cnt={model.rollout_step_counter}",
                      flush=True)
        # —— 检测: 每个仿真步实时跑 ——
        for kind, o in mon.update(t, ee, objpos):
            trig = kind in ("DROP", "STALL")
            events.append((fi, kind, o))
            print(f"[ONLINE] t={t:3d}  {kind} {o or ''}{'  <<< 失败' if trig else ''}", flush=True)
            if args.recover and trig and pending is None and n_recover < args.max_recover:
                pending = (kind, o)               # 不立刻打断, 先挂起
                events.append((fi, "PENDING", o))
                rem = model.multistep - model.rollout_step_counter
                print(f"[PENDING] t={t} 失败已记录, 先把当前动作块剩余 {rem} 步执行完再重规划", flush=True)
        # —— 调度: 当前动作块刚执行完(块边界) 且 有挂起失败 -> 此刻才停小脑、调大脑 ——
        # --interrupt: 不等块执行完, 检测到立刻打断
        at_boundary = (model.rollout_step_counter == 0) or args.interrupt
        if pending is not None and at_boundary and not done:
            kind, o = pending; pending = None
            fobj = o or next((x for x in grasp_objs if not mon.placed[x]), None)
            print(f"[BOUNDARY] t={t} 动作块执行完, 停止小脑, 交给大脑重规划", flush=True)
            events.append((len(frames) - 1, "CHUNK_DONE", fobj))
            cur_plan_new, rec_tag, elapsed = None, None, 0.0
            if brain is not None:
                # 监控给的客观事实: 每个物体在不在它目标里 + 离目标距离(供大脑判断, 别靠图瞎猜)
                fact_lines = []
                for ob in grasp_objs:
                    p = objpos.get(ob); tn = targets.get(ob); tp = objpos.get(tn) if tn else None
                    if p is None:
                        continue
                    if tp is not None:
                        d = float(np.linalg.norm(p[:2] - tp[:2]))
                        where = f"INSIDE {tn}" if d < 0.09 else f"on the table, NOT in {tn}"
                        fact_lines.append(f"- {ob}: {where} (distance to {tn} = {d:.2f} m)")
                facts = "\n".join(fact_lines)
                failed_plan = subplans.get(fobj, "")
                t0 = time.time()
                dec = brain.decide(frames[-1], task_i.language, plan_text, f"{kind} {o or ''}",
                                   grasp_objs, mon.placed, facts, failed_plan)
                elapsed = time.time() - t0
                print(f"[BRAIN] thought for {elapsed:.1f}s", flush=True)
                print(f"[BRAIN] diagnosis: {dec['diagnosis']}", flush=True)
                print(f"[BRAIN] recoverable={dec['recoverable']} reason: {dec['reason']}", flush=True)
                if dec.get("recoverable") and dec.get("plan"):
                    cur_plan_new = dec["plan"]
                    print(f"[RECOVER] #{n_recover + 1} qwen plan: {cur_plan_new[:90]}...", flush=True)
            if cur_plan_new is None:   # 启发式 / 千问不可恢复 -> 回退子计划
                tgt = next((o2 for o2 in grasp_objs if not mon.placed[o2]), None)
                if tgt and tgt in subplans:
                    cur_plan_new = subplans[tgt]; rec_tag = tgt
                    print(f"[RECOVER] #{n_recover + 1} fallback -> {tgt}", flush=True)
            # —— 时延建模: 大脑思考期间小脑停发新动作, 机械臂保持当前位姿, 仿真照常推进 ——
            # 千问用实测耗时; 否则用 --brain_delay_s 人为设定(扫时延曲线, 与真大脑解耦)
            delay_s = elapsed if brain is not None else args.brain_delay_s
            hold = int(round(delay_s * FPS))
            if hold > 0:
                ee_before = np.array(obs.get("robot0_eef_pos", [0, 0, 0])).copy()
                obj_before = objpos.get(fobj, np.zeros(3)).copy()
                s0 = len(frames); hold_act = np.array([0., 0., 0., 0., 0., 0., float(action[6])])
                # --freeze_arm: 暂停期间把机械臂关节强行锁死(完全不动), 只让物体随物理下落
                rob_q, rob_v, saved_q = [], [], None
                if args.freeze_arm:
                    for j in range(env.sim.model.njnt):
                        nm = env.sim.model.joint_id2name(j)
                        if nm and ("robot0" in nm or "gripper" in nm):
                            rob_q.append(env.sim.model.jnt_qposadr[j]); rob_v.append(env.sim.model.jnt_dofadr[j])
                    saved_q = env.sim.data.qpos[rob_q].copy()
                for _ in range(hold):
                    obs, _, done, _ = env.step(hold_act)
                    if args.freeze_arm:   # 每步把机械臂关节复位回暂停前, 速度清零 -> 真冻结
                        env.sim.data.qpos[rob_q] = saved_q
                        env.sim.data.qvel[rob_v] = 0.0
                        env.sim.forward()
                        try:
                            obs = env._get_observations(force_update=True)
                        except Exception:
                            pass
                    frames.append(obs["agentview_image"][::-1].copy())
                    if done:
                        break
                ee_after = np.array(obs.get("robot0_eef_pos", [0, 0, 0]))
                obj_after = env.sim.data.body_xpos[bid[fobj]].copy() if fobj in bid else np.zeros(3)
                print(f"[HOLD] {hold}步暂停: 机械臂末端漂移={np.linalg.norm(ee_after - ee_before):.4f}m "
                      f"(z {ee_before[2]:.3f}->{ee_after[2]:.3f})  物体{fobj}漂移="
                      f"{np.linalg.norm(obj_after - obj_before):.4f}m", flush=True)
                events.append((s0, "THINKING", f"{delay_s:.1f}s"))
                think_spans.append((s0, len(frames)))
            if cur_plan_new is not None:
                # 新计划给小脑 + reset(清空旧动作块缓冲和观测历史) -> 下一步用新计划重新扩散推理
                cur_plan = cur_plan_new; model.reset(); n_recover += 1
                events.append((len(frames) - 1, "RECOVER", rec_tag))
                if args.dump_tag:   # 把小脑恢复后即将看到的 obs(两路相机图+夹爪)dump 出来对比
                    gq = np.array(obs.get("robot0_gripper_qpos", [0, 0]))
                    av = obs["agentview_image"][::-1]; wr = obs["robot0_eye_in_hand_image"][::-1]
                    both = np.concatenate([av, wr], axis=1)
                    fn = f"/root/autodl-tmp/recovery_obs_{args.dump_tag}_{n_recover}.png"
                    cv2.imwrite(fn, cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
                    print(f"[DUMP] {fn}  夹爪qpos={gq}  目标{fobj}位置={objpos.get(fobj)}  新计划={cur_plan[:60]}",
                          flush=True)
        if done:
            print(f"[closed] task done at t={t}")
            break
    env.close()
    print(f"[closed] success={bool(done)}  placed={mon.placed}  recoveries={n_recover}")
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = None
    tag = f"{'recover' if args.recover else 'norecover'}_{args.brain if args.recover else 'none'}"
    out = os.path.join("/root/autodl-tmp", f"closed_loop_{tag}.mp4")
    vid = compose_video(frames, events, inj_win, bool(done), out, ff, think_spans)
    print(f"[closed] SAVED video = {vid}")


if __name__ == "__main__":
    main()
