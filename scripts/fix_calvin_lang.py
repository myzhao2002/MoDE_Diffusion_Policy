#!/usr/bin/env python
"""
修复 VyoJ 子集合并后的语言标注。

问题: VyoJ 每个子集里的 lang_annotations/auto_lang_ann.npy 是 **整集 ABCD_D** 的标注
(每子集 ~22966 条, 引用全 24 子集的帧), 且没有 lang_filtered/。naive 合并 N 个子集 =>
N×22966 条, 其中大量引用的帧不在我们下载的子集里 => 训练 _load_episode 会 FileNotFoundError。

正解: 以"该标注的整个窗口 [start, end] 的帧是否都在合并目录的 episode 文件里"为准过滤,
并按 (start,end) 去重, 得到 ~N×(子集帧占比) 条有效标注。写成 lang_filtered/ 和修正的
lang_annotations/, 这样 lang_folder=lang_filtered 训练即可(plan 任务标签也读这个 task 字段)。

用法:
  python scripts/fix_calvin_lang.py --train /root/autodl-tmp/CALVIN-datasets/calvin_abcd_6sub/training
"""
import argparse, os, glob
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="合并后的 training 目录")
    ap.add_argument("--src_folder", default="lang_annotations",
                    help="读取整集标注的来源子目录(子集里就叫 lang_annotations)")
    args = ap.parse_args()
    train = Path(args.train)

    # 1) 现有帧号集合
    have = set()
    for p in glob.glob(str(train / "episode_*.npz")):
        have.add(int(os.path.basename(p)[8:15]))
    print(f"[fix] 现有 episode 帧 = {len(have)}  范围 {min(have)}~{max(have)}")

    # 2) 载入(可能被 naive 合并污染的)整集标注
    src = train / args.src_folder / "auto_lang_ann.npy"
    d = np.load(src, allow_pickle=True).item()
    indx = list(d["info"]["indx"])
    lang = d["language"]
    n0 = len(indx)
    print(f"[fix] 源标注数 = {n0}")

    # 3) 先按 (start,end) 去重(N 子集叠的重复整集), 再按"整窗帧全在 have"过滤
    seen = set()
    keep_idx = []
    for i, se in enumerate(indx):
        s, e = int(se[0]), int(se[1])
        key = (s, e)
        if key in seen:
            continue
        seen.add(key)
        # 整窗帧都要在(disk_dataset 会逐帧 load)
        if all((f in have) for f in range(s, e + 1)):
            keep_idx.append(i)
    print(f"[fix] 去重后唯一 = {len(seen)}  其中整窗帧齐全(有效) = {len(keep_idx)}")

    # 4) 按 keep_idx 同步过滤所有并行数组
    new = {"info": {}, "language": {}}
    # info: indx 必过滤; 其它 info 键若与标注同长也过滤, 否则原样
    new["info"]["indx"] = [indx[i] for i in keep_idx]
    for k, v in d["info"].items():
        if k == "indx":
            continue
        if hasattr(v, "__len__") and len(v) == n0:
            new["info"][k] = ([v[i] for i in keep_idx] if not isinstance(v, np.ndarray)
                              else v[keep_idx])
        else:
            new["info"][k] = v
    for k, v in lang.items():
        if isinstance(v, np.ndarray) and v.shape[0] == n0:
            new["language"][k] = v[keep_idx]
        elif hasattr(v, "__len__") and len(v) == n0:
            new["language"][k] = [v[i] for i in keep_idx]
        else:
            new["language"][k] = v  # 标量/不同长 原样
    nkeep = len(new["info"]["indx"])
    print(f"[fix] 写出有效标注 = {nkeep}  (emb shape={getattr(new['language'].get('emb',None),'shape',None)})")

    # 5) 写 lang_annotations(覆盖) + lang_filtered(训练默认读它)
    for folder in ["lang_annotations", "lang_filtered"]:
        outd = train / folder
        outd.mkdir(parents=True, exist_ok=True)
        np.save(outd / "auto_lang_ann.npy", new, allow_pickle=True)
        print(f"[fix] 写入 {outd/'auto_lang_ann.npy'}")

    # 6) 自检
    bad = sum(1 for se in new["info"]["indx"]
              if int(se[0]) not in have or int(se[1]) not in have)
    print(f"[fix] 复检: 起止帧缺失的标注 = {bad}/{nkeep}  -> {'OK' if bad==0 else '仍有问题!'}")


if __name__ == "__main__":
    main()
