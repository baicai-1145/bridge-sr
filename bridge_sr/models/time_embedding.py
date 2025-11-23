from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """将标量时间 t ∈ [0,1] 映射到高维向量。

    使用正余弦位置编码 + 两层 MLP。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:
        t: shape (B,) or (B,1), assumed in [0,1].

        Returns:
        time_emb: shape (B, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half_dim, device=device).float()
            / half_dim
        )
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            pad = self.dim - emb.size(-1)
            emb = torch.nn.functional.pad(emb, (0, pad))
        return self.mlp(emb)


