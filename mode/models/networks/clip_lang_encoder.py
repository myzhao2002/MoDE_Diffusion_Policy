"""
CLIP 语言编码器 — 将文本(任务指令 / 原子计划)编码为 512-d 向量

本项目使用两种 CLIP 封装:
  - LangClip: 基于 OpenAI CLIP RN50,支持对 plan 按 "->" 切分后分段编码再平均池化
  - LangClip2: 基于 OpenAI CLIP ViT-B/32,直接全句编码

数据流 (LangClip, 用于 plan 编码):
  plan_text = "Approach(soup) -> Grasp(soup) -> Lift(soup) -> ..."
       ↓ split by "->"
  ["Approach(soup)", "Grasp(soup)", "Lift(soup)", ...]
       ↓ 逐段 CLIP 编码
  [512-d, 512-d, 512-d, ...]
       ↓ mean pooling
  单个 512-d 全局 plan 向量

冻结: 默认 freeze_backbone=True,CLIP 参数不参与训练
"""

from typing import List

import torch
import torch.nn as nn
import clip
from mode.models.networks.clip import build_model, load_clip, tokenize
from transformers import (
    AutoProcessor,
    AutoModel,
    SiglipProcessor,
    SiglipModel
)


class LangClip(nn.Module):
    """
    CLIP RN50 语言编码器 — 支持 plan 分段编码 + 平均池化。

    对于原子计划字符串 "A -> B -> C",会先按 "->" 切分成子动作,
    分别 CLIP 编码后取 mean,得到融合所有子动作语义的单一向量。
    输出: [B, 1, 512]
    """
    def __init__(self, freeze_backbone: bool = True, model_name: str = "RN50"):
        super(LangClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading language CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        """加载 CLIP 模型权重"""
        model, _ = load_clip(model_name, device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        self.output_dim = 512

    def forward(self, x: List) -> torch.Tensor:
        """
        编码一批文本(支持 plan 按 "->" 切分)。

        Args:
            x: 文本列表, 每个元素是一个 plan 字符串如 "Approach(x) -> Grasp(x) -> ..."
        Returns:
            [B, 1, 512] 编码后的 plan embedding
        """
        with torch.no_grad():
            batch_embs = []
            for text in x:
                # 按 "->" 切分成独立子动作
                sub_tasks = [t.strip() for t in text.split("->") if t.strip()]
                # 逐子动作 CLIP 编码
                tokens = tokenize(sub_tasks).to(self.device).long()
                sub_embs = self.clip_rn50.encode_text(tokens)  # [N_subtasks, 512]
                # 平均池化 → 1 个 512-d 全局 plan 向量
                fused_emb = sub_embs.mean(dim=0, keepdim=True)  # [1, 512]
                batch_embs.append(fused_emb)
            emb = torch.cat(batch_embs, dim=0)  # [B, 512]
        return torch.unsqueeze(emb, 1)  # [B, 1, 512]

    @torch.no_grad()
    def encode_plan_steps(self, text: str) -> torch.Tensor:
        """把单条 plan 按 "->" 切成步骤, 每步独立 CLIP 编码, 不做池化。

        用于实验四 v2 (方案三): 规划作为一串 token 序列(每步 1 token)当
        交叉注意力的 KV, 让模型靠注意力对齐到"当前执行到第几步"。

        Args:
            text: 单条 plan 字符串 "Approach(x) -> Grasp(x) -> ..."
        Returns:
            [S, 512] 每步一个 CLIP 向量(S = 子动作数)
        """
        sub_tasks = [t.strip() for t in text.split("->") if t.strip()]
        if len(sub_tasks) == 0:
            sub_tasks = [text.strip() or "."]
        tokens = tokenize(sub_tasks).to(self.device).long()
        return self.clip_rn50.encode_text(tokens)  # [S, 512]


class LangClip2(nn.Module):
    """
    CLIP ViT-B/32 语言编码器 — 全句直接编码(不做 plan 切分)。

    用于简单的任务指令编码,输出 [B, 512]。
    """
    def __init__(self, freeze_backbone: bool = True, model_name: str = "ViT-B/32") -> None:
        super().__init__()
        clip_model, clip_preprocess = clip.load(model_name)
        if freeze_backbone:
            for _, param in clip_model.named_parameters():
                param.requires_grad = False

        self.text_tokenizer = clip.tokenize
        self.text_encoder = clip_model

    def forward(self, x: List) -> torch.Tensor:
        """
        全句编码一批文本。
        Args:
            x: 文本列表
        Returns:
            [B, 512] 文本 embedding
        """
        inputs = self.text_tokenizer(x)
        device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder.encode_text(inputs.to(device))
        return encoder_hidden_states

