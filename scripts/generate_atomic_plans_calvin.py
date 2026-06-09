"""
Batch-generate atomic action plans for CALVIN tasks using the SAME local
Qwen-VL OpenAI-compatible endpoint used for LIBERO (autodl vLLM @ localhost:8000).

Adapted from generate_atomic_plans_qwen.py (LIBERO). Key differences:
- Task source: CALVIN `auto_lang_ann.npy` (task labels + language annotations),
  NOT per-task HDF5 files. One plan per UNIQUE CALVIN task label (~34), keyed by
  the task label so the CALVIN dataloader can look it up.
- Representative image: rgb_static from the first frame of a representative
  episode of each task (episode_XXXXXXX.npz).
- Prompt keeps the same DSL + parsing, plus CALVIN-style templates/examples
  (push block / move slider / toggle led / rotate block).

用法 / Usage (autodl, Qwen-VL 服务要先起来):
    conda activate lerobot
    # 确认本地 Qwen 服务在跑: curl http://localhost:8000/v1/models
    python scripts/generate_atomic_plans_calvin.py \
      --calvin_train_dir /root/autodl-tmp/CALVIN-datasets/calvin_vyoj/training \
      --base_url http://localhost:8000/v1 --model qwen-vl-local \
      --out /root/autodl-tmp/CALVIN-datasets/plans_calvin.jsonl
"""

import os
for v in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(v, None)

import argparse
import base64
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import httpx
from openai import OpenAI


VALID_DSL_ACTIONS = {
    "Approach", "Align", "Grasp", "Lift", "Transport", "Lower", "Release", "Retreat",
    "Push", "Pull", "Rotate", "Twist", "Slide", "Tilt",
}

JSON_OUTPUT_SUFFIX = """

## ADDITIONAL OUTPUT: STRUCTURED JSON (fallback, must match Phase 2 exactly)
After your Phase 1 + Phase 2 output above, append a JSON code block:

```json
{
  "plan": [
    {"action": "Approach", "args": ["blue_block"]},
    {"action": "Push",     "args": ["blue_block", "right"]},
    {"action": "Retreat",  "args": []}
  ]
}
```

Strict JSON rules:
- "action" must match the DSL exactly (case-sensitive).
- "args" is always a list (use [] for actions with no args, like Retreat).
- Object names in snake_case.
- **The JSON must contain EXACTLY the same atoms as Phase 2 — same count, same order.**
- Wrap in ```json ... ``` code fences.
""".strip()


def build_planning_prompt(selected_task: str, p0_info: str) -> str:
    base_prompt = f"""
You are an Embodied AI World Model and Robot Task Planner for a TABLETOP scene
(CALVIN: a table with colored blocks, a sliding cabinet, a drawer, a button, a
switch, and an LED/lightbulb).
Your job is to infer a physically plausible robot plan from an initial image.

## 1. CONTEXT
- Goal Task: "{selected_task}"
- Initial Robot State: {p0_info}

## 2. DSL DEFINITION (Robot Atomic Action DSL v0.1)
### Core Actions
- Approach(object), Align(object, pose_hint), Grasp(object), Lift(object)
- Transport(object, target), Lower(object, target), Release(object), Retreat()
### Interaction Actions
- Push(object, direction), Pull(object, direction)
- Rotate(object, axis), Twist(object, direction)
- Slide(object, direction), Tilt(object, target_or_direction)

## 3. CRITICAL PLANNING RULES

### Rule A: Count sub-goals carefully — DO NOT INVENT
A "sub-goal" is one **independent physical action** explicitly stated in the task.
- Single-action verb = 1 sub-goal: "push the block right", "open the drawer",
  "turn on the led", "move the slider left", "lift the block".
- Multi-action verbs joined by "and" = 2+ sub-goals.
- A pick-and-place ("put X in Y", "place X on Y", "stack X on Y") is ONE sub-goal.
**Do NOT add sub-goals not stated in the task.**

### Rule B: Object reference consistency
- Drawer: grasp "drawer_handle", move "drawer". Slider/cabinet: "slider".
- Button/switch/led/lightbulb: "<x>_switch" (e.g. led_switch, lightbulb_switch).
- Blocks: "<color>_block" (red_block, blue_block, pink_block).
- Approach/Align target the HANDLE/SWITCH, not the parent object.
- Use the SAME object name throughout one sub-goal sequence.

### Rule C: Action templates
- Free object pick-and-place (1 sub-goal, 8 atoms):
  Approach(obj) -> Align(obj, grasp_pose) -> Grasp(obj) -> Lift(obj) ->
  Transport(obj, target) -> Lower(obj, target) -> Release(obj) -> Retreat()
- Push a block on the table (1 sub-goal, 4 atoms, NO grasp):
  Approach(obj) -> Align(obj, push_pose) -> Push(obj, <direction>) -> Retreat()
- Move slider / cabinet (1 sub-goal, 4 atoms):
  Approach(slider) -> Align(slider, push_pose) -> Slide(slider, <direction>) -> Retreat()
- Open / close drawer (1 sub-goal, 6 atoms):
  Approach(drawer_handle) -> Align(drawer_handle, pull_pose) -> Grasp(drawer_handle) ->
  Pull(drawer, <backward|forward>) -> Release(drawer_handle) -> Retreat()
- Toggle button / switch / led / lightbulb (1 sub-goal, 4 atoms):
  Approach(<x>_switch) -> Align(<x>_switch, press_pose) -> Push(<x>_switch, down) -> Retreat()
- Rotate a block in place (1 sub-goal, 6 atoms):
  Approach(obj) -> Align(obj, grasp_pose) -> Grasp(obj) -> Rotate(obj, z) ->
  Release(obj) -> Retreat()
- Lift a block off the table (1 sub-goal, 5 atoms):
  Approach(obj) -> Align(obj, grasp_pose) -> Grasp(obj) -> Lift(obj) -> Retreat()

## 4. PHASE 1: SUB-GOAL COUNTING
Step 1: Count the number of independent action verbs in the task.
Step 2: List each sub-goal explicitly.

## 5. PHASE 2: ACTION DERIVATION
Concatenate sequences for each sub-goal in order.
**The Phase 2 [Begin]...[End] line and the JSON MUST contain exactly the same atoms.**

## 6. EXAMPLES

Example A (push block right): "push the blue block to the right"
Phase 1: 1 sub-goal — push blue_block to the right.
Phase 2: [Begin] --- [Approach(blue_block)] --- [Align(blue_block, push_pose)] --- [Push(blue_block, right)] --- [Retreat()] --- [End]

Example B (move slider): "move the slider to the left"
Phase 1: 1 sub-goal — slide the slider left.
Phase 2: [Begin] --- [Approach(slider)] --- [Align(slider, push_pose)] --- [Slide(slider, left)] --- [Retreat()] --- [End]

Example C (open drawer): "open the drawer"
Phase 1: 1 sub-goal — open the drawer.
Phase 2: [Begin] --- [Approach(drawer_handle)] --- [Align(drawer_handle, pull_pose)] --- [Grasp(drawer_handle)] --- [Pull(drawer, backward)] --- [Release(drawer_handle)] --- [Retreat()] --- [End]

Example D (toggle led): "turn on the led light"
Phase 1: 1 sub-goal — toggle the led switch on.
Phase 2: [Begin] --- [Approach(led_switch)] --- [Align(led_switch, press_pose)] --- [Push(led_switch, down)] --- [Retreat()] --- [End]

Example E (rotate block): "rotate the pink block to the right"
Phase 1: 1 sub-goal — rotate pink_block.
Phase 2: [Begin] --- [Approach(pink_block)] --- [Align(pink_block, grasp_pose)] --- [Grasp(pink_block)] --- [Rotate(pink_block, z)] --- [Release(pink_block)] --- [Retreat()] --- [End]

Example F (lift block): "lift the red block from the table"
Phase 1: 1 sub-goal — lift red_block.
Phase 2: [Begin] --- [Approach(red_block)] --- [Align(red_block, grasp_pose)] --- [Grasp(red_block)] --- [Lift(red_block)] --- [Retreat()] --- [End]

## 7. OUTPUT FORMAT
Phase 1: <count> sub-goal(s) — <list each>
Phase 2: [Begin] --- [Action(param)] --- ... --- [End]
""".strip()
    return base_prompt + "\n\n" + JSON_OUTPUT_SUFFIX


# ---- parsing (verbatim from the LIBERO version, keeps identical behavior) ----
def parse_phase2_to_plan(llm_output: str) -> dict:
    m = re.search(r"Phase\s*2\s*:\s*\[Begin\]([\s\S]*?)\[End\]", llm_output)
    if not m:
        raise ValueError("No Phase 2 [Begin]...[End] block found")
    body = m.group(1)
    pattern = r"\[(\w+)\s*\(([^\[\]]*?)\)\s*\]"
    matches = re.findall(pattern, body)
    plan = []
    for action, args_str in matches:
        args = []
        if args_str.strip():
            for a in args_str.split(","):
                a = a.strip().strip('"').strip("'")
                if a:
                    args.append(a)
        plan.append({"action": action, "args": args})
    if not plan:
        raise ValueError("No valid actions found in Phase 2 body")
    return {"plan": plan}


def parse_plan_output(llm_output: str) -> dict:
    if not llm_output:
        raise ValueError("Empty output")
    try:
        return parse_phase2_to_plan(llm_output)
    except Exception as e_phase2:
        fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output)
        if fence:
            json_str = fence.group(1)
        else:
            fb = re.search(r"\{[\s\S]*?\"plan\"[\s\S]*?\]\s*\}", llm_output)
            if not fb:
                raise ValueError(f"Both Phase 2 and JSON parsing failed. Phase 2 err: {e_phase2}")
            json_str = fb.group(0)
        return json.loads(json_str)


def validate_plan(plan):
    if "plan" not in plan or not isinstance(plan["plan"], list):
        return False, "Missing or invalid 'plan'"
    for i, step in enumerate(plan["plan"]):
        if "action" not in step or step["action"] not in VALID_DSL_ACTIONS:
            return False, f"Step {i}: invalid action '{step.get('action')}'"
        if "args" not in step or not isinstance(step["args"], list):
            return False, f"Step {i}: invalid args"
    return True, f"OK ({len(plan['plan'])} steps)"


def plan_to_text(plan_obj: dict) -> str:
    parts = []
    for step in plan_obj.get("plan", []):
        act = step.get("action", "")
        args = step.get("args", [])
        parts.append(f"{act}({', '.join(args)})" if args else f"{act}()")
    return " -> ".join(parts)


def load_calvin_tasks(train_dir: Path):
    """Return list of (task_label, instruction_text, rep_frame_index) for each
    UNIQUE CALVIN task label, read from training/lang_annotations/auto_lang_ann.npy."""
    ann_path = train_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        ann_path = train_dir / "auto_lang_ann.npy"
    data = np.load(ann_path, allow_pickle=True).item()
    tasks = data["language"]["task"]
    anns = data["language"]["ann"]
    indx = data["info"]["indx"]  # list of (start, end) frame ranges, parallel to tasks/anns

    seen = {}
    for i, t in enumerate(tasks):
        if t in seen:
            continue
        start = int(indx[i][0])
        seen[t] = (str(anns[i]), start)
    out = [(t, instr, start) for t, (instr, start) in seen.items()]
    print(f"[CALVIN] {len(tasks)} annotations, {len(out)} unique task labels")
    return out


def load_rep_image_b64(train_dir: Path, frame_index: int):
    """Load rgb_static from episode_{frame_index:07d}.npz and return base64 jpeg."""
    npz_path = train_dir / f"episode_{frame_index:07d}.npz"
    arr = np.load(npz_path, allow_pickle=True)
    img = arr["rgb_static"]  # (H, W, 3) uint8, upright (no flip for CALVIN)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


def run_one_task(client, model_name, train_dir, task_label, instruction, frame_index,
                 p0_info="No initial pose data."):
    b64 = load_rep_image_b64(train_dir, frame_index)
    prompt = build_planning_prompt(instruction, p0_info)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=1500,
        temperature=0.1,
    )
    latency = time.time() - t0
    raw = resp.choices[0].message.content

    result = {
        "task": task_label,           # KEY used by the CALVIN dataloader
        "instruction": instruction,
        "rep_frame": frame_index,
        "latency": latency,
        "raw": raw,
    }
    try:
        plan = parse_plan_output(raw)
        ok, msg = validate_plan(plan)
        result["parse_ok"] = ok
        result["parse_msg"] = msg
        if ok:
            result["plan"] = plan
            result["plan_text"] = plan_to_text(plan)
        else:
            result["error"] = msg
    except Exception as e:
        result["parse_ok"] = False
        result["parse_msg"] = str(e)
        result["error"] = str(e)

    usage = getattr(resp, "usage", None)
    if usage is not None:
        result["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        result["completion_tokens"] = getattr(usage, "completion_tokens", None)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calvin_train_dir", type=str,
                    default="/root/autodl-tmp/CALVIN-datasets/calvin_vyoj/training")
    ap.add_argument("--model", type=str, default="qwen-vl-local")
    ap.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--out", type=str,
                    default="/root/autodl-tmp/CALVIN-datasets/plans_calvin.jsonl")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    train_dir = Path(args.calvin_train_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0, connect=10.0))
    client = OpenAI(api_key="EMPTY", base_url=args.base_url, http_client=http_client)

    tasks = load_calvin_tasks(train_dir)
    success = failed = 0
    with out_path.open("w", encoding="utf-8") as wf:
        for i, (task_label, instruction, frame_index) in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task_label} :: {instruction}")
            try:
                rec = run_one_task(client, args.model, train_dir, task_label, instruction, frame_index)
            except Exception as e:
                rec = {"task": task_label, "instruction": instruction, "rep_frame": frame_index,
                       "parse_ok": False, "parse_msg": str(e), "error": str(e)}
            if rec.get("parse_ok"):
                success += 1
                print(f"  ✅ {rec.get('parse_msg')} | {rec.get('plan_text','')}")
            else:
                failed += 1
                print(f"  ❌ {rec.get('parse_msg')}")
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()
            time.sleep(args.sleep)

    print("\n===== Summary =====")
    print(f"Output file: {out_path.resolve()}")
    print(f"Success: {success} | Failed: {failed} | Total: {success + failed}")


if __name__ == "__main__":
    main()
