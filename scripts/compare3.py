"""三路闭环视频并排: 无恢复 / 快反射恢复 / 千问恢复。真实仿真, 短的末帧定格补齐。
  python scripts/compare3.py --v1 demo_i0_norecover.mp4 --v2 demo_i0_reflex.mp4 --v3 demo_i0_qwen.mp4 --out demo_i0_3way.mp4
标题在脚本里设(避免 shell 拆空格)。"""
import argparse
import os
import subprocess
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLES = ["NO RECOVERY (stuck)", "FAST REFLEX (~0 delay)", "QWEN BRAIN (4s pause)"]
COLORS = [(40, 40, 130), (40, 110, 40), (20, 90, 150)]


def read_all(path):
    cap = cv2.VideoCapture(path)
    fs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    cap.release()
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--out", default="demo_3way.mp4")
    ap.add_argument("--dir", default="/root/autodl-tmp")
    args = ap.parse_args()

    vids = [read_all(os.path.join(args.dir, v)) for v in (args.v1, args.v2, args.v3)]
    n = max(len(v) for v in vids)
    vids = [v + [v[-1]] * (n - len(v)) for v in vids]   # 末帧定格补齐
    h = vids[0][0].shape[0]; w = vids[0][0].shape[1]
    th = 30; gap = 8
    W = w * 3 + gap * 2; H = h + th
    tmp = os.path.join(args.dir, args.out.replace(".mp4", "_raw.mp4"))
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
    for i in range(n):
        canvas = np.full((H, W, 3), 20, np.uint8)
        for k in range(3):
            x0 = k * (w + gap)
            f = vids[k][i]
            canvas[th:th + f.shape[0], x0:x0 + f.shape[1]] = f
            cv2.rectangle(canvas, (x0, 0), (x0 + w, th), COLORS[k], -1)
            cv2.putText(canvas, TITLES[k], (x0 + 8, 21), FONT, 0.5, (255, 255, 255), 1)
        wr.write(canvas)
    wr.release()
    out = os.path.join(args.dir, args.out)
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                       check=True, capture_output=True, timeout=180)
        os.remove(tmp)
    except Exception as e:
        print("ffmpeg convert failed:", e); out = tmp
    print("SAVED", out, "frames", n)


if __name__ == "__main__":
    main()
