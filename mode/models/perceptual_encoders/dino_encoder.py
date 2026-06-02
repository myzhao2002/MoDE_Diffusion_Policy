"""
Frozen DINOv2 visual encoder.

Replaces the FiLM-ResNet frontend. Returns dense patch tokens
(`forward_features(x)['x_norm_patchtokens']`) instead of a single pooled
vector, so a downstream cross-attention / Perceiver head can let the language
plan "query" the patches.

Notes
-----
* DINOv2 expects ImageNet normalization, but the MoDE data pipeline normalizes
  with CLIP mean/std (see conf/datamodule/transforms/libero_transforms.yaml).
  We therefore RE-NORMALIZE inside this module (undo CLIP norm -> apply
  ImageNet norm). This keeps the transform config untouched and the ResNet
  baseline path unaffected.
* DINOv2 ViT patch size is 14; input H/W must be divisible by 14. LIBERO uses
  static=224 (16x16=256 patches) and gripper=112 (8x8=64 patches), both OK.
* The backbone is frozen; only the downstream Perceiver heads train.
* First run needs internet to fetch the hub repo + weights (on autodl enable
  `/etc/network_turbo`). Afterwards it is cached under TORCH_HOME.
"""
import torch
import torch.nn as nn


# CLIP normalization stats used by the existing data transforms.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# ImageNet stats expected by DINOv2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        clip_norm_input: bool = True,
    ):
        super().__init__()
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", model_name, trust_repo=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 '{model_name}' via torch.hub. The first "
                f"run needs internet (on autodl run `source /etc/network_turbo`). "
                f"Original error: {e}"
            )

        self.embed_dim = self.model.embed_dim  # 384 (S), 768 (B), 1024 (L)
        self.clip_norm_input = clip_norm_input

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._frozen = freeze

        # Re-normalization buffers (registered so .to(device/dtype) moves them).
        self.register_buffer("clip_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1))
        self.register_buffer("in_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval mode regardless of the parent module.
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] image batch, CLIP-normalized (per the data pipeline).
        Returns:
            patch tokens [B, N_patches, embed_dim]
        """
        if self.clip_norm_input:
            dt = x.dtype
            # undo CLIP normalization -> [0,1], then apply ImageNet normalization
            x = x * self.clip_std.to(dt) + self.clip_mean.to(dt)
            x = (x - self.in_mean.to(dt)) / self.in_std.to(dt)

        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            feats = self.model.forward_features(x)
        return feats["x_norm_patchtokens"]
