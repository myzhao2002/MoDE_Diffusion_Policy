"""
Step 3 段级语义对齐:切好的每个段抽 3 帧(start/mid/end)给 Qwen-VL(阿里云百炼),
让它从 DSL 原子动作里选一个标注 + 识别对象,纠正规则切分的语义错标。
默认只跑一条 demo(省 token)。
"""
import os, glob, json, base64, time, re, argparse
for v in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(v, None)
import numpy as np
import h5py
import cv2
import httpx
from openai import OpenAI

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # 用前: export DASHSCOPE_API_KEY=sk-xxx
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DSL = (
    "Approach(object), Align(object,pose_hint), Grasp(object), Lift(object), "
    "Transport(object,target), Lower(object,target), Release(object), Retreat(), "
    "Push(object,direction), Pull(object,direction), Rotate(object,axis), "
    "Twist(object,direction), Slide(object,direction), Tilt(object,target)"
)
VALID = {"Approach", "Align", "Grasp", "Lift", "Transport", "Lower", "Release",
         "Retreat", "Push", "Pull", "Rotate", "Twist", "Slide", "Tilt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", default="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it")
    ap.add_argument("--model", default="qwen-vl-max-latest")
    ap.add_argument("--out", default="/root/autodl-tmp/seg_labels_qwen.json")
    args = ap.parse_args()

    root = "/root/autodl-tmp/LIBERO-datasets"
    h5 = [p for p in glob.glob(f"{root}/{args.suite}/*.hdf5") if args.task in p][0]
    f = h5py.File(h5, "r"); d = f["data"]["demo_0"]
    imgs = np.array(d["obs"]["agentview_rgb"])
    grip = np.array(d["actions"])[:, 6]
    ee_z = np.array(d["obs"]["ee_pos"])[:, 2]
    T = len(imgs)

    segj = [p for p in glob.glob(f"{root}/segments_v3/{args.suite}/*.json")
            if args.task in p and "checkpoint" not in p][0]
    sj = json.load(open(segj))
    segs = sj["demos"]["demo_0"]["segments"]
    meta = sj["meta"]
    objects = meta.get("objects", [])
    instruction = meta.get("instruction", args.task)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL,
                    http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0, connect=10.0)))

    def b64(t):
        img = imgs[int(t)][::-1]                       # flip_ud
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode()

    def label(seg):
        s = int(seg["start"]); e = int(min(seg["end"], T - 1)); mid = (s + e) // 2
        g0, g1 = grip[s], grip[min(e, len(grip) - 1)]
        gstate = ("closing/grasping" if g0 < 0 and g1 > 0 else
                  "opening/releasing" if g0 > 0 and g1 < 0 else
                  "closed/holding" if g1 > 0 else "open")
        dz = ee_z[min(e, len(ee_z) - 1)] - ee_z[s]
        zmot = "moving up" if dz > 0.02 else "moving down" if dz < -0.02 else "roughly level"
        prompt = f"""You are a robot motion analyst.
Robot task: "{instruction}"
Objects in scene: {', '.join(objects)}

Below are 3 frames (start, middle, end) of ONE motion segment of the gripper.
Objective low-level motion: gripper is {gstate}; end-effector {zmot}.

Classify this segment as EXACTLY ONE atomic action from this DSL:
{DSL}

Rules:
- Look at WHERE the gripper actually is and WHAT it touches (do not just guess from the task name).
- If it presses/turns a mechanism (button/knob), the object is e.g. stove_knob, drawer_handle.
- "action" must be one DSL name exactly. "args" is a list (snake_case), [] for Retreat.
Output ONLY JSON: {{"action":"...","args":["..."],"reason":"one short sentence"}}"""
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(t)}"}} for t in (s, mid, e)]
        resp = client.chat.completions.create(model=args.model,
                                              messages=[{"role": "user", "content": content}],
                                              max_tokens=300, temperature=0.1)
        raw = resp.choices[0].message.content
        m = re.search(r"\{[\s\S]*\}", raw)
        return json.loads(m.group(0)) if m else {"action": "?", "args": [], "reason": raw[:80]}

    print(f"task: {instruction}")
    print(f"objects: {objects}\n")
    print(f"{'seg':<4}{'[frames]':<12}{'RULE label':<34}{'QWEN label':<30}reason")
    print("-" * 110)
    results = []
    for i, seg in enumerate(segs):
        s, e = seg["start"], seg["end"]
        rule = f"{seg['action']}({','.join(seg.get('args', []) or [])})"
        try:
            q = label(seg)
            ql = f"{q.get('action')}({','.join(q.get('args', []) or [])})"
            ok = q.get("action") in VALID
            print(f"{i+1:<4}[{s}-{e}]".ljust(16) + f"{rule:<34}{ql:<30}{str(q.get('reason',''))[:42]}")
            results.append({"seg": i + 1, "frames": [s, e], "rule": rule, "qwen": q})
        except Exception as ex:
            print(f"{i+1:<4}[{s}-{e}]  {rule:<34}ERROR: {ex}")
            results.append({"seg": i + 1, "frames": [s, e], "rule": rule, "error": str(ex)})
        time.sleep(0.3)
    json.dump({"task": args.task, "instruction": instruction, "segments": results},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nSAVED {args.out}")


if __name__ == "__main__":
    main()
