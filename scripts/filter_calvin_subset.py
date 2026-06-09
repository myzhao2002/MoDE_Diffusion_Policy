"""
Filter a VyoJ CALVIN subset so MoDE's CALVIN dataloader can use it.

Problem: each VyoJ subset ships the FULL auto_lang_ann.npy (global frame indices
spanning the whole ABCD_D), but only a SLICE of episode_*.npz files is present.
So most language annotations point to episodes that don't exist in this subset →
MoDE's lang dataset would crash at load time.

This script rewrites auto_lang_ann.npy keeping ONLY the annotations whose full
[start, end] frame range exists on disk, and saves it under a NEW lang folder
(default: lang_filtered) so the original is untouched. Train with
`lang_folder=lang_filtered`.

用法 / Usage:
    python scripts/filter_calvin_subset.py \
      --root /root/autodl-tmp/CALVIN-datasets/calvin_vyoj \
      --splits training validation --out_folder lang_filtered
"""
import argparse
from pathlib import Path

import numpy as np


def present_frames(split_dir: Path) -> set:
    return set(
        int(p.name.split("_")[1].split(".")[0])
        for p in split_dir.glob("episode_*.npz")
    )


def filter_split(split_dir: Path, out_folder: str) -> int:
    present = present_frames(split_dir)
    if not present:
        print(f"[skip] {split_dir} has no episode_*.npz")
        return 0

    src = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not src.exists():
        src = split_dir / "auto_lang_ann.npy"
    d = np.load(src, allow_pickle=True).item()
    indx = d["info"]["indx"]

    keep = []
    for i, (s, e) in enumerate(indx):
        s, e = int(s), int(e)
        if s in present and e in present and all(j in present for j in range(s, e + 1)):
            keep.append(i)

    out = {"info": {}, "language": {}}
    for k, v in d["info"].items():
        out["info"][k] = [v[i] for i in keep] if k == "indx" else v
    for k, v in d["language"].items():
        if isinstance(v, np.ndarray):
            out["language"][k] = v[keep]
        else:
            out["language"][k] = [v[i] for i in keep]

    dst_dir = split_dir / out_folder
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / "auto_lang_ann.npy", out)
    print(f"[{split_dir.name}] kept {len(keep)}/{len(indx)} annotations "
          f"({len(present)} frames present) -> {dst_dir/'auto_lang_ann.npy'}")
    return len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,
                    default="/root/autodl-tmp/CALVIN-datasets/calvin_vyoj")
    ap.add_argument("--splits", type=str, nargs="+", default=["training", "validation"])
    ap.add_argument("--out_folder", type=str, default="lang_filtered")
    args = ap.parse_args()

    root = Path(args.root)
    total = 0
    for sp in args.splits:
        d = root / sp
        if d.exists():
            total += filter_split(d, args.out_folder)
        else:
            print(f"[skip] {d} does not exist")
    print(f"\nDone. Total kept annotations: {total}")
    print(f"Train with: lang_folder={args.out_folder}")


if __name__ == "__main__":
    main()
