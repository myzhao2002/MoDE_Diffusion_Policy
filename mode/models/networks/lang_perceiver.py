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
