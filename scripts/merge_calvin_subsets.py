#!/usr/bin/env python
"""
合并 VyoJ/calvin-ABCD-D-subsets 的多个训练子集为一个 MoDE 可吃的 training/ 目录。

每个子集自包含: training/{episode_*.npz, ep_start_end_ids.npy, ep_lens.npy,
lang_annotations/auto_lang_ann.npy, lang_filtered/auto_lang_ann.npy, ...}。
HulcDataModule 只吃单个 root_data_dir(training/+validation/),所以要把 N 个子集:
  - episode_*.npz  -> os.rename 移动进合并目录(同盘秒移、不占额外空间; 全局帧号唯一不冲突)
  - ep_start_end_ids.npy -> vstack
  - lang_annotations / lang_filtered 的 auto_lang_ann.npy -> 并接 info.indx + language.{ann,emb,task}
  - statistics.yaml / scene_info.npy / .hydra -> 从第一个子集拷贝
validation 直接复用现有 vyoj/validation,不在本脚本处理。

用法:
  python scripts/merge_calvin_subsets.py \
    --stage /root/autodl-tmp/CALVIN-datasets/_subsets_raw \
    --out   /root/autodl-tmp/CALVIN-datasets/calvin_abcd_6sub/training
"""
import argparse, os, glob, shutil
from pathlib import Path
import numpy as np


def find_train_dir(subset_root: Path) -> Path:
    """子集解压后 episode 可能在 subset_root 或 subset_root/training 下。"""
    cands = [subset_root, subset_root / "training"]
    for c in cands:
        if c.is_dir() and len(list(c.glob("episode_*.npz"))) > 0:
            return c
    # 再往里找一层
    for sub in subset_root.rglob("training"):
        if sub.is_dir() and len(list(sub.glob("episode_*.npz"))) > 0:
            return sub
    raise FileNotFoundError(f"在 {subset_root} 下找不到含 episode_*.npz 的 training 目录")


def merge_lang(dst_dir: Path, src_dirs, sub_name: str):
    """合并某个 lang 子目录(lang_annotations / lang_filtered)的 auto_lang_ann.npy。"""
    merged = None
    n_files = 0
    for sd in src_dirs:
        f = sd / sub_name / "auto_lang_ann.npy"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True).item()
        n_files += 1
        if merged is None:
            merged = d
            # 转成可扩展的容器
            merged["info"]["indx"] = list(d["info"]["indx"])
            for k in list(d["language"].keys()):
                v = d["language"][k]
                merged["language"][k] = list(v) if not isinstance(v, np.ndarray) else v
            continue
        # info.indx
        merged["info"]["indx"].extend(list(d["info"]["indx"]))
        # language.*
        for k, v in d["language"].items():
            if isinstance(v, np.ndarray):
                merged["language"][k] = np.concatenate([np.asarray(merged["language"][k]), v], axis=0)
            else:
                merged["language"][k].extend(list(v))
    if merged is None:
        print(f"  [lang] {sub_name}: 无文件, 跳过")
        return
    # emb 等转回 ndarray
    if "emb" in merged["language"] and not isinstance(merged["language"]["emb"], np.ndarray):
        merged["language"]["emb"] = np.asarray(merged["language"]["emb"])
    out = dst_dir / sub_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "auto_lang_ann.npy", merged, allow_pickle=True)
    n_ann = len(merged["info"]["indx"])
    print(f"  [lang] {sub_name}: 合并 {n_files} 个子集 -> {n_ann} 条标注 -> {out/'auto_lang_ann.npy'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="子集解压区(含 subset_training_*)")
    ap.add_argument("--out", required=True, help="输出合并后的 training 目录")
    ap.add_argument("--copy", action="store_true", help="拷贝 episode 而非移动(默认移动, 省空间)")
    args = ap.parse_args()

    stage = Path(args.stage)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subset_roots = sorted([p for p in stage.glob("subset_training_*") if p.is_dir()])
    if not subset_roots:
        raise SystemExit(f"{stage} 下没有 subset_training_* 目录")
    print(f"[merge] 发现 {len(subset_roots)} 个子集: {[p.name for p in subset_roots]}")

    train_dirs = []
    all_ep_se = []
    moved, skipped = 0, 0
    for sr in subset_roots:
        td = find_train_dir(sr)
        train_dirs.append(td)
        # 1) episodes
        for npz in td.glob("episode_*.npz"):
            dst = out / npz.name
            if dst.exists():
                skipped += 1
                continue
            if args.copy:
                shutil.copy2(npz, dst)
            else:
                os.rename(npz, dst)  # 同盘秒移
            moved += 1
        # 2) ep_start_end_ids
        se_f = td / "ep_start_end_ids.npy"
        if se_f.exists():
            all_ep_se.append(np.load(se_f))
        print(f"  [{sr.name}] train_dir={td}  episodes moved so far={moved} skipped={skipped}")

    # 写合并的 ep_start_end_ids
    if all_ep_se:
        ep_se = np.vstack(all_ep_se)
        np.save(out / "ep_start_end_ids.npy", ep_se)
        print(f"[merge] ep_start_end_ids: {ep_se.shape[0]} 条序列 -> {out/'ep_start_end_ids.npy'}")

    # 3) lang 两套
    merge_lang(out, train_dirs, "lang_annotations")
    merge_lang(out, train_dirs, "lang_filtered")

    # 4) 标量/配置从第一个子集拷
    first = train_dirs[0]
    for name in ["statistics.yaml", "scene_info.npy", "ep_lens.npy", ".hydra"]:
        src = first / name
        dst = out / name
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  [misc] 拷 {name} (来自 {first.name})")

    n_ep = len(list(out.glob("episode_*.npz")))
    print(f"\n[merge] 完成。合并目录 {out}")
    print(f"  episode_*.npz = {n_ep}")
    print(f"  下一步: 把它当 root_data_dir 的 training/，validation/ 软链或复用 vyoj/validation。")
    print(f"  例: ln -s /root/autodl-tmp/CALVIN-datasets/calvin_vyoj/validation {out.parent}/validation")


if __name__ == "__main__":
    main()
