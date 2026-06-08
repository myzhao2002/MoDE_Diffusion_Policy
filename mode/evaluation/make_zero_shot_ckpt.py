"""
把 OXE 预训练权重(MoDE_Pretrained, safetensors warm-start 格式)转换成一个
Lightning `.ckpt` + `.hydra/config.yaml`,以便直接用 mode_evaluate_libero.py
评测「微调前(zero-shot)」的预训练模型在 LIBERO 上的表现。

Convert the OXE-pretrained warm-start weights (the MoDE_Pretrained safetensors
dir) into a Lightning `.ckpt` plus a `.hydra/config.yaml`, so the standard
mode_evaluate_libero.py can evaluate the *pre-finetune* (zero-shot) pretrained
model on LIBERO.

为什么需要它 / Why this is needed
--------------------------------
eval 用 Lightning 的 `load_from_checkpoint`,要求一个 `.ckpt` 文件 + 训练 run 的
`.hydra/config.yaml`;而 MoDE_Pretrained 是 `model_cleaned.safetensors`(走的是
MoDEAgent.load_pretrained_parameters 那条 warm-start 路径),不能直接喂给 eval。
本脚本先用和训练完全一致的方式建模并 warm-start,然后落成 eval 能读的格式。

⚠️ 重要说明 / IMPORTANT CAVEAT
-----------------------------
OXE 预训练模型没见过 LIBERO,动作归一化/本体/相机都不同,**zero-shot 成功率预计
极低(接近 0),这是预期现象,不能据此判定原文造假**。真正的复现是「用预训练权重
在 LIBERO 上微调」—— 实验一 Baseline 已得到 90.4 ≈ 原文 92,复现其实已经成立。
本脚本只是把「微调前 vs 微调后」的对照量化出来(衡量微调贡献),不是打假判据。

用法 / Usage (在 autodl 上 / on autodl)
---------------------------------------
    conda activate lerobot
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/MoDE_Diffusion_Policy/LIBERO:/root/autodl-tmp/MoDE_Diffusion_Policy

    # 1) 转换:warm-start 预训练权重 -> Lightning ckpt
    python mode/evaluation/make_zero_shot_ckpt.py \
      model.start_from_pretrained=True \
      model.ckpt_path=/root/autodl-tmp/MoDE_Pretrained \
      +out_dir=/root/autodl-tmp/MoDE_ckpts/zero_shot_pretrained

    # 2) 评测:用现有 eval 脚本评 zero-shot 模型(先评单套件 libero_10 快速看)
    python mode/evaluation/mode_evaluate_libero.py \
      train_folder=/root/autodl-tmp/MoDE_ckpts/zero_shot_pretrained \
      checkpoint=/root/autodl-tmp/MoDE_ckpts/zero_shot_pretrained/zero_shot.ckpt \
      plan_file=null \
      benchmark_name=libero_10
"""
import os
from pathlib import Path

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf


@hydra.main(config_path="../../conf", config_name="config_libero")
def main(cfg):
    out_dir = Path(cfg.get("out_dir", "/root/autodl-tmp/MoDE_ckpts/zero_shot_pretrained"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 用和训练完全一致的方式建模;MoDEAgent.__init__ 内部在 start_from_pretrained=True
    #    且 ckpt_path 指向 MoDE_Pretrained 时,会调用 load_pretrained_parameters 完成
    #    OXE 权重 warm-start(ResNet 视觉 + DiT 主干等)。
    print("[zero-shot] instantiating model & warm-starting from pretrained ...")
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    # 2) 写一份 .hydra/config.yaml 供 eval 重建模型;把 start_from_pretrained 关掉,
    #    避免 eval 加载时再次 warm-start(多余且慢),也解除对 MoDE_Pretrained 目录的
    #    依赖(eval 直接从下面这个 ckpt 的 state_dict 恢复完整权重)。
    save_cfg = cfg.copy()
    OmegaConf.set_struct(save_cfg, False)
    save_cfg.model.start_from_pretrained = False
    hydra_dir = out_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    OmegaConf.save(save_cfg, hydra_dir / "config.yaml")

    # 3) 存成 Lightning `load_from_checkpoint` 能读的 ckpt(state_dict + 超参)。
    #    eval 会用 .hydra/config.yaml 的 model 配置重建同构模型,再严格加载这里的
    #    state_dict —— 二者结构一致,strict load 通过。
    ckpt = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": [],
    }
    ckpt_path = out_dir / "zero_shot.ckpt"
    torch.save(ckpt, ckpt_path)

    print(f"[zero-shot] saved Lightning ckpt -> {ckpt_path}")
    print(f"[zero-shot] saved hydra config  -> {hydra_dir / 'config.yaml'}")
    print("[zero-shot] 下一步用 mode_evaluate_libero.py 评测(见脚本顶部 Usage)。")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
