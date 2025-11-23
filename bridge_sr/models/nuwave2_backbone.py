from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .time_embedding import TimeEmbedding


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        time_dim: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.time_proj = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        time_emb: (B, time_dim)
        """
        h = self.conv(x)
        t = self.time_proj(time_emb).unsqueeze(-1)
        h = h + t
        h = self.act(h)
        h = self.res_conv(h)
        return x + h


class BridgeSRBackbone(nn.Module):
    """简化版 NU-Wave2 风格骨干网络。

    输入:
      - x_t: (B, T)
      - x_T: (B, T)
      - t: (B,)
    输出:
      - x_hat_0: (B, T)
    """

    def __init__(
        self,
        channels: int = 64,
        resblocks: int = 20,
        kernel_size: int = 3,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.resblocks = resblocks

        self.time_embedding = TimeEmbedding(time_dim)

        # 两个通道：当前状态 x_t 与条件 x_T
        self.input_conv = nn.Conv1d(2, channels, kernel_size=7, padding=3)

        blocks = []
        dilations = []
        for i in range(resblocks):
            d = 2 ** (i % 8)  # 循环的指数膨胀
            dilations.append(d)
            blocks.append(
                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    time_dim=time_dim,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x_t: (B, T) or (B, 1, T)
          t: (B,)
          x_T: (B, T) or (B, 1, T)
        Returns:
          x_hat_0: (B, T)
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        if x_T.dim() == 2:
            x_T = x_T.unsqueeze(1)

        x = torch.cat([x_t, x_T], dim=1)
        h = self.input_conv(x)

        time_emb = self.time_embedding(t)
        for block in self.blocks:
            h = block(h, time_emb)

        out = self.output_conv(h)
        out = out.squeeze(1)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


