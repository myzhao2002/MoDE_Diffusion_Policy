"""
Language-conditioned Perceiver resampler.

A small set of learnable latents, biased by the (whole-plan) language embedding,
cross-attend over DINO patch tokens. This is the "language queries the patches"
mechanism: the output is a fixed number `num_latents` of visual tokens that
focus on plan-relevant image regions, regardless of how many patches the
backbone produced.

Design follows the Perceiver-Resampler used in Flamingo
(lucidrains/flamingo-pytorch), with two changes:
  * the latents are additively conditioned on a language vector (the plan),
  * an input projection maps the media (patch) dim to the working dim, so the
    backbone (e.g. DINOv2 768-d) and the policy token dim (512) can differ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       media (patch) tokens [B, N, dim]
            latents: query latents        [B, K, dim]
        Returns:
            [B, K, dim]
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        # latents attend to both the media AND themselves (Flamingo style).
        kv_input = torch.cat([x, latents], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LangPerceiver(nn.Module):
    def __init__(
        self,
        media_dim: int,
        dim: int = 512,
        num_latents: int = 4,
        depth: int = 2,
        cond_dim: int = 512,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.media_proj = nn.Linear(media_dim, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.lang_proj = nn.Linear(cond_dim, dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, media: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            media: patch tokens [B, N, media_dim]
            cond:  language (plan) embedding [B, cond_dim]
        Returns:
            visual tokens [B, num_latents, dim]
        """
        b = media.shape[0]
        x = self.media_proj(media)

        if cond.dim() == 3:
            cond = cond.squeeze(1)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        latents = latents + self.lang_proj(cond).unsqueeze(1)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


# =====================================================================
# 实验四 v2 (方案三): 视觉查询分步规划 (Visual queries the per-step plan)
# ---------------------------------------------------------------------
# 与上面的 LangPerceiver 相比, 三处本质区别:
#   1. Q/KV 翻转: Q = 当前视觉(自适应池化后的 patch), K,V = 分步规划 token。
#      这样注意力天然就是"当前画面对齐到规划第几步"的进度信号。
#   2. 取消固定可学 latent: query 数量由视觉池化网格(query_grid)决定,
#      参数无关, 不再被 nn.Parameter 的 num_latents 绑死。
#   3. 纯交叉注意力(不再 Flamingo 式 cat[media, latents] 混自注意力),
#      FFN 换成 SwiGLU, 规划 token 加可学绝对位置编码。
# 输出维度仍 = media_dim(2048), 复用 warm-started tok_emb -> 零接口侵入。
# =====================================================================


class SwiGLUFeedForward(nn.Module):
    """门控 FFN (Shazeer 2020, LLaMA/PaLM 同款)。

    隐藏维 ×2/3 补偿: SwiGLU 有 3 个权重矩阵, 乘 2/3 后总参数量与
    普通两层 FFN(mult=4)对齐。
    """
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.norm = nn.LayerNorm(dim)
        self.w_in = nn.Linear(dim, hidden * 2, bias=False)   # 一次出 gate + value
        self.w_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * value)            # SiLU(x)=x·sigmoid(x)


class VisualQueryCrossAttention(nn.Module):
    """Q = 视觉 token, K,V = 分步规划 token。纯交叉注意力 + KV padding mask。"""
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor,
                kv_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q_tokens: 视觉 query [B, Q, dim]
            kv_tokens: 规划 token [B, S, dim]
            kv_mask:  规划有效位 [B, S] (1=有效步, 0=padding), 可选
        Returns:
            [B, Q, dim]
        """
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)

        h = self.heads
        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if kv_mask is not None:
            mask = kv_mask[:, None, None, :].to(torch.bool)   # [B,1,1,S]
            sim = sim.masked_fill(~mask, torch.finfo(sim.dtype).min)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class VisualPlanResampler(nn.Module):
    """方案三主体: 视觉(自适应池化后)查询分步规划。

    Args:
        media_dim: 视觉 patch 维度(ResNet50 = 2048), 也是输出维度。
        plan_dim:  单步规划 CLIP 向量维度(512)。
        dim:       注意力内部工作维度(Flamingo 风格, 默认 512)。
        query_grid: 视觉自适应池化网格边长; query 数 = grid^2(2 -> 4 个)。
        depth:     交叉注意力层数。
        max_plan_steps: 规划步数上限(可学位置编码表大小)。
    """
    def __init__(
        self,
        media_dim: int,
        plan_dim: int = 512,
        dim: int = 512,
        query_grid: int = 2,
        depth: int = 1,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        max_plan_steps: int = 16,
    ):
        super().__init__()
        self.query_grid = query_grid
        self.num_queries = query_grid * query_grid
        self.max_plan_steps = max_plan_steps

        self.media_proj = nn.Linear(media_dim, dim)
        self.plan_proj = nn.Linear(plan_dim, dim)
        self.plan_pos = nn.Embedding(max_plan_steps, dim)   # 规划步骤位置编码

        self.layers = nn.ModuleList([
            nn.ModuleList([
                VisualQueryCrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                SwiGLUFeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, media_dim)           # 升回 2048, 接 tok_emb

    def forward(self, media: torch.Tensor, plan_tokens: torch.Tensor,
                plan_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            media: ResNet layer4 patch tokens [B, N, media_dim] (N=49 或 16)
            plan_tokens: 分步规划 token [B, S, plan_dim]
            plan_mask:  规划有效位 [B, S] (1=有效), 可选
        Returns:
            plan-aligned 视觉 token [B, query_grid^2, media_dim]
        """
        b, n, _ = media.shape
        side = int(round(n ** 0.5))
        # [B, N, C] -> [B, C, side, side] -> 自适应平均池化到 grid×grid -> [B, grid^2, C]
        m = rearrange(media, "b (hh ww) c -> b c hh ww", hh=side, ww=side)
        q = F.adaptive_avg_pool2d(m, output_size=self.query_grid)
        q = rearrange(q, "b c hh ww -> b (hh ww) c")
        q = self.media_proj(q)                              # [B, Q, dim]

        # 规划 token: 截断到上限 -> 投影 -> 加位置编码
        s = plan_tokens.shape[1]
        if s > self.max_plan_steps:
            plan_tokens = plan_tokens[:, :self.max_plan_steps]
            if plan_mask is not None:
                plan_mask = plan_mask[:, :self.max_plan_steps]
            s = self.max_plan_steps
        p = self.plan_proj(plan_tokens)                     # [B, S, dim]
        pos = self.plan_pos(torch.arange(s, device=p.device))
        p = p + pos.unsqueeze(0)

        for attn, ff in self.layers:
            q = attn(q, p, kv_mask=plan_mask) + q
            q = ff(q) + q

        return self.out_proj(self.norm(q))                  # [B, Q, media_dim]
