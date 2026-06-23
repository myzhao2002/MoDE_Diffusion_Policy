"""把两个闭环视频并排合成一个对比视频(真实仿真, 无造假帧)。
左: 无恢复 -> 掉落后卡死失败;  右: 有恢复 -> 检测+重抓成功。
短的那个用末帧定格补齐, 这样右边成功后定格、左边还在挣扎, 对比明显。

  python scripts/compare_sidebyside.py --left closed_loop_norecover_none.mp4 \
    --right closed_loop_recover_heuristic.mp4 --out compare.mp4 \
    --left_title "NO RECOVERY -> FAIL" --right_title "WITH RECOVERY -> SUCCESS"
"""
import argparse
import os
import subprocess
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", default="compare.mp4")
    ap.add_argument("--left_title", default="NO RECOVERY")
    ap.add_argument("--right_title", default="WITH RECOVERY")
    ap.add_argument("--dir", default="/root/autodl-tmp")
    args = ap.parse_args()

    L = read_all(os.path.join(args.dir, args.left))
    R = read_all(os.path.join(args.dir, args.right))
    n = max(len(L), len(R))
    # 末帧定格补齐
    L += [L[-1]] * (n - len(L))
    R += [R[-1]] * (n - len(R))
    h = max(L[0].shape[0], R[0].shape[0])
    w = L[0].shape[1]
    th = 30          # 顶部标题条
    gap = 8          # 中间分隔
    W = w * 2 + gap
    H = h + th
    tmp = os.path.join(args.dir, args.out.replace(".mp4", "_raw.mp4"))
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
    for i in range(n):
        canvas = np.full((H, W, 3), 20, np.uint8)
        lf, rf = L[i], R[i]
        canvas[th:th + lf.shape[0], :lf.shape[1]] = lf
        canvas[th:th + rf.shape[0], w + gap:w + gap + rf.shape[1]] = rf
        cv2.rectangle(canvas, (0, 0), (w, th), (40, 40, 130), -1)
        cv2.rectangle(canvas, (w + gap, 0), (W, th), (40, 110, 40), -1)
        cv2.putText(canvas, args.left_title, (10, 21), FONT, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, args.right_title, (w + gap + 10, 21), FONT, 0.6, (255, 255, 255), 2)
        wr.write(canvas)
    wr.release()
    out = os.path.join(args.dir, args.out)
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                       check=True, capture_output=True, timeout=120)
        os.remove(tmp)
    except Exception as e:
        print("ffmpeg convert failed:", e); out = tmp
    print("SAVED", out, "frames", n)


if __name__ == "__main__":
    main()
