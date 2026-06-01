import torch
import torch.nn as nn


class TaskPlanFuser(nn.Module):
    """
    Fuse task text embedding and atomic-plan text embedding into one goal embedding.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, task_emb: torch.Tensor, plan_emb: torch.Tensor) -> torch.Tensor:
        # Supports [B, D] or [B, 1, D]
        if task_emb.dim() == 3:
            task_emb = task_emb.squeeze(1)
        if plan_emb.dim() == 3:
            plan_emb = plan_emb.squeeze(1)

        # Align input dtype with the fuser weights. During eval the MoDE model is
        # cast to half precision, but CLIP text embeddings stay float32, which
        # would raise "mat1 and mat2 must have the same dtype" in the Linear layer.
        target_dtype = self.net[0].weight.dtype
        task_emb = task_emb.to(target_dtype)
        plan_emb = plan_emb.to(target_dtype)

        fused = self.net(torch.cat([task_emb, plan_emb], dim=-1))
        return fused.unsqueeze(1)

