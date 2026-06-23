"""
千问 VL 决策大脑(进程内 transformers, Qwen3-VL-2B)。
在闭环里: 监控检测到失败(DROP/STALL)时, 把当前 agentview 图 + 上下文喂给它,
它看图判断"哪个物体掉在桌上需要重抓重放 / 还是放弃", 返回决策。

CLI 自测(给一张图 + 任务 + 物体列表, 看它怎么决策):
  python scripts/qwen_brain.py --image /root/autodl-tmp/drop_frame.png \
    --task "put both the alphabet soup and the tomato sauce in the basket" \
    --objects alphabet_soup,tomato_sauce
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import torch
from PIL import Image

QWEN_PATH = "/root/autodl-tmp/models/Qwen3-VL-2B-Instruct"


class QwenBrain:
    def __init__(self, model_path=QWEN_PATH, device="cuda:0"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
        print("[brain] Qwen3-VL loaded.")

    @torch.no_grad()
    def decide(self, image_rgb, task, full_plan, event, objects, placed, facts="", failed_plan=""):
        """看图做结构化诊断 + 生成式重规划(通用, 不假设目标是篮子)。
        full_plan: 原始原子计划文本。facts: 监控给的客观传感器事实(物体在不在目标里/坐标)。
        failed_plan: 失败所在的那段原子子计划(让模型在失败点附近局部续接, 不整段重来)。
        返回 dict{diagnosis, recoverable, plan, reason, raw}。"""
        img = Image.fromarray(image_rgb)
        done = [o for o in objects if placed.get(o, False)]
        todo = [o for o in objects if not placed.get(o, False)]
        facts_block = (
            f"Objective sensor facts (from the robot's position sensors - these are GROUND TRUTH, "
            f"trust them over the image if they conflict):\n{facts}\n" if facts else "")
        failed_block = (
            f"The failure happened while executing THIS part of the plan:\n{failed_plan}\n"
            if failed_plan else "")
        prompt = (
            "You are a robot execution supervisor. You see the robot's current camera image.\n"
            f"Task: {task}\n"
            f"Original atomic plan:\n{full_plan}\n"
            f"An execution monitor just detected a failure: {event}.\n"
            f"{failed_block}"
            f"{facts_block}"
            f"Sub-goals already completed (do NOT redo these): {done}. Still to complete: {todo}.\n"
            "Combine the sensor facts with the image to judge what actually went wrong and the current scene.\n"
            "Output ONLY a JSON object (no other text) with these fields:\n"
            '  "diagnosis": one short sentence on what went wrong, based on the facts and image;\n'
            '  "recoverable": true or false (can the robot still finish the task?);\n'
            '  "plan": a SHORT recovery plan to execute NEXT, as atomic actions joined by " -> ". '
            "RESUME LOCALLY at the failure: redo ONLY the failed sub-goal (re-grasp and place the object "
            "that was dropped/failed), then continue with the still-to-complete sub-goals. Do NOT restart "
            "sub-goals already completed. Use ONLY the action types and object/target names that appear in "
            "the original plan (do NOT invent names, do NOT assume the destination is a basket - use "
            "whatever the original plan uses);\n"
            '  "reason": one short sentence explaining the plan.\n'
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inputs = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw = self.proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        import json, re
        diagnosis, plan, reason, recov = "", None, "", True
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                diagnosis = str(d.get("diagnosis", "")); plan = d.get("plan")
                reason = str(d.get("reason", "")); recov = bool(d.get("recoverable", True))
            except Exception:
                pass
        return {"diagnosis": diagnosis, "recoverable": recov, "plan": plan, "reason": reason, "raw": raw}


DEFAULT_PLAN = ("Approach(alphabet_soup) -> Align(alphabet_soup, grasp_pose) -> Grasp(alphabet_soup) -> "
                "Lift(alphabet_soup) -> Transport(alphabet_soup, basket) -> Lower(alphabet_soup, basket) -> "
                "Release(alphabet_soup) -> Retreat() -> Approach(tomato_sauce) -> Align(tomato_sauce, grasp_pose) -> "
                "Grasp(tomato_sauce) -> Lift(tomato_sauce) -> Transport(tomato_sauce, basket) -> "
                "Lower(tomato_sauce, basket) -> Release(tomato_sauce) -> Retreat()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", default="put both the alphabet soup and the tomato sauce in the basket")
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--event", default="DROP alphabet_soup (object slipped out of gripper)")
    ap.add_argument("--objects", default="alphabet_soup,tomato_sauce")
    args = ap.parse_args()
    import numpy as np
    img = np.array(Image.open(args.image).convert("RGB"))
    objs = args.objects.split(",")
    brain = QwenBrain()
    res = brain.decide(img, args.task, args.plan, args.event, objs, {o: False for o in objs})
    print("[brain] === RAW ===\n", res["raw"])
    print("[brain] diagnosis :", res["diagnosis"])
    print("[brain] recoverable:", res["recoverable"])
    print("[brain] plan      :", res["plan"])
    print("[brain] reason    :", res["reason"])


if __name__ == "__main__":
    main()
