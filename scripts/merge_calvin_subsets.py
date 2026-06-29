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


def filter_and_write_lang(dst_dir: Path, src_dirs, have: set):
    """合并 + 过滤语言标注, 写 lang_annotations/ 与 lang_filtered/。

    ⚠️ VyoJ 子集里的 lang_annotations 是 **整集 ABCD_D** 标注(每子集都含全量,引用全 24
    子集的帧), 且没有 lang_filtered/。所以不能 naive 拼接(会得到 N×全量 + 大量引用缺失帧)。
    正确做法: 汇总所有子集的 lang_annotations -> 按 (start,end) 去重 -> 只保留"整窗 [start,end]
    的帧都在合并目录(have)里"的标注, 同步过滤并行的 language.{ann,emb,task}。
    """
    # 1) 汇总(去重)所有子集的整集标注
    base = None
    seen = set()
    keep_se = []       # 去重后的 (start,end)
    src_records = []   # 对应 (源dict, 行号) 以便取并行字段
    for sd in src_dirs:
        f = sd / "lang_annotations" / "auto_lang_ann.npy"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True).item()
        if base is None:
            base = d
        for i, se in enumerate(d["info"]["indx"]):
            s, e = int(se[0]), int(se[1])
            if (s, e) in seen:
                continue
            seen.add((s, e))
            keep_se.append((d, i, s, e))
    if base is None:
        print("  [lang] 子集无 lang_annotations, 跳过"); return
    # 2) 只保留整窗帧齐全的
    valid = [(d, i, s, e) for (d, i, s, e) in keep_se
             if all((fr in have) for fr in range(s, e + 1))]
    print(f"  [lang] 去重唯一={len(seen)}  整窗齐全(有效)={len(valid)}")
    # 3) 重建 dict(并行字段对齐)
    new = {"info": {"indx": [(s, e) for (_, _, s, e) in valid]}, "language": {}}
    lang_keys = list(base["language"].keys())
    for k in lang_keys:
        col = []
        is_np = isinstance(base["language"][k], np.ndarray)
        for (d, i, _, _) in valid:
            col.append(d["language"][k][i])
        new["language"][k] = np.asarray(col) if is_np else col
    # 4) 写两个文件夹(训练默认读 lang_filtered;task 字段供 plan 匹配)
    for folder in ["lang_annotations", "lang_filtered"]:
        o = dst_dir / folder
        o.mkdir(parents=True, exist_ok=True)
        np.save(o / "auto_lang_ann.npy", new, allow_pickle=True)
    bad = sum(1 for (s, e) in new["info"]["indx"] if s not in have or e not in have)
    print(f"  [lang] 写出 {len(valid)} 条 -> lang_annotations + lang_filtered  (缺失复检={bad})")


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

    # 3) lang: 过滤整集标注到合并目录实际存在的帧(去重 + 整窗在场), 写 lang_annotations + lang_filtered
    have = set()
    for npz in out.glob("episode_*.npz"):
        have.add(int(npz.name[8:15]))
    filter_and_write_lang(out, train_dirs, have)

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
