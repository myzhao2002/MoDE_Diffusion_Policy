"""
独立 CALVIN 单序列评测 + 录视频脚本。
加载已训好的 CALVIN ckpt,跑 1 条 5-子任务序列,把全程 rollout 录成 mp4。

用法 / Usage:
    python scripts/calvin_record_one.py \
      --train_folder /root/autodl-tmp/MoDE_ckpts/calvin_exp4_20260609_135626 \
      --ckpt /root/autodl-tmp/MoDE_ckpts/calvin_exp4_20260609_135626/saved_models/last.ckpt \
      --plan_file /root/autodl-tmp/CALVIN-datasets/plans_calvin.jsonl \
      --root_data_dir /root/autodl-tmp/CALVIN-datasets/calvin_vyoj \
      --save_dir /root/autodl-tmp/calvin_videos
"""
import argparse
import os
import sys
from pathlib import Path

# repo paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from mode.evaluation.utils import load_pl_module_from_checkpoint


def find_ckpt(folder: Path):
    """If --ckpt is a dir, pick last.ckpt or the newest *.ckpt under it."""
    if folder.is_file():
        return folder
    last = folder / "saved_models" / "last.ckpt"
    if last.exists():
        return last
    cks = sorted(folder.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not cks:
        raise FileNotFoundError(f"no .ckpt under {folder}")
    return cks[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_folder", required=True,
                    help="hydra run dir of the calvin training (含 .hydra/config.yaml)")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint 路径(可以是 .ckpt 文件或包含 saved_models/ 的目录)")
    ap.add_argument("--plan_file", required=True)
    ap.add_argument("--root_data_dir", required=True)
    ap.add_argument("--save_dir", default="/root/autodl-tmp/calvin_videos")
    ap.add_argument("--num_sequences", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    seed_everything(0, workers=True)

    train_folder = Path(args.train_folder)
    ckpt_path = find_ckpt(Path(args.ckpt))
    print(f"[record-one] train_folder = {train_folder}")
    print(f"[record-one] ckpt          = {ckpt_path}")

    # 1) 加载训练时的 hydra 配置(模型超参一致才能 load_from_checkpoint)
    cfg = OmegaConf.load(train_folder / ".hydra" / "config.yaml")

    # 2) 重建模型 + 加载训好的权重
    print("[record-one] loading model ...")
    model = load_pl_module_from_checkpoint(ckpt_path, config_dir=str(train_folder))
    model = model.to(args.device).eval()
    model.freeze()
    print(f"[record-one] model on {args.device}, dtype={model.dtype}")

    # 3) 重建一个 dataset(rollout 只用它拿 abs_datasets_dir / lang_folder,不真训)
    cfg.datamodule.root_data_dir = args.root_data_dir
    # HulcDataModule.setup() 自己用 datasets_dir 显式 instantiate datasets,所以这里
    # 必须 _recursive_=False,否则 hydra 会先尝试 instantiate dataset、缺 datasets_dir。
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(cfg.datamodule, False)
    cfg.datamodule._recursive_ = False
    datamodule = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    datamodule.setup()
    val_dataloaders = datamodule.val_dataloader()
    dataset = (val_dataloaders["lang"] if isinstance(val_dataloaders, dict)
               else val_dataloaders[0]).dataset

    # 4) 构造 rollout callback(num_videos=1 / log_video_to_file=True / 1 序列)
    rollout_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "conf/callbacks/rollout_lh/calvin.yaml"
    )
    rollout_cfg.num_videos = args.num_sequences
    rollout_cfg.num_sequences = args.num_sequences
    rollout_cfg.skip_epochs = 0
    rollout_cfg.rollout_freq = 1
    rollout_cfg.log_video_to_file = True
    rollout_cfg.save_dir = args.save_dir
    rollout_cfg.plan_file = args.plan_file
    rollout_cfg.lang_folder = cfg.lang_folder

    os.makedirs(args.save_dir, exist_ok=True)
    # rollout 内部 hydra.utils.instantiate 需要 conf 顶层 hydra 注册;直接 instantiate 即可
    rollout = hydra.utils.instantiate(rollout_cfg)

    # 5) 模拟 trainer 把 dataloaders 暴露给 callback
    class _FakeTrainer:
        callbacks = []
        val_dataloaders = val_dataloaders
        def __init__(self): pass

    fake_trainer = _FakeTrainer()
    # 6) 触发 rollout(on_validation_start 内做 env 创建 + rollout)
    print(f"[record-one] starting {args.num_sequences} rollout(s) with video recording ...")
    rollout.on_validation_start(fake_trainer, model)
    results = rollout.evaluate_policy(model)
    print(f"[record-one] rollout results (subtasks completed per sequence): {results}")
    # 7) flush videos
    if rollout.rollout_video is not None:
        rollout.rollout_video.write_to_tmp()

    print(f"\n[record-one] videos saved under: {args.save_dir}")
    for p in sorted(Path(args.save_dir).rglob("*.mp4")):
        print("  ", p)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
