from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class BridgeGMaxConfig:
    beta_0: float
    beta_1: float
    t_min: float = 0.0
    t_max: float = 1.0
    num_grid_points: int = 1000


class BridgeGMaxSchedule:
    """Bridge-g_max 噪声调度与 σ_t^2 / σ̄_t^2 查表。

    假设 f(t) = 0，因此 α_t = 1。
    我们在 [0, 1] 上构建一个固定时间网格，通过数值积分预计算：
      - σ_t^2 = ∫_0^t g^2(τ) dτ
      - σ̄_t^2 = ∫_t^1 g^2(τ) dτ
    供训练和采样阶段插值使用。
    """

    def __init__(
        self,
        cfg: BridgeGMaxConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._build_table()

    def _build_table(self) -> None:
        n = self.cfg.num_grid_points
        t_grid = torch.linspace(0.0, 1.0, n, dtype=self.dtype, device=self.device)
        g2_grid = self.g2(t_grid)

        # 梯形积分，计算 σ_t^2
        sigma2_grid = torch.zeros_like(t_grid)
        dt = t_grid[1:] - t_grid[:-1]
        trapezoid = 0.5 * (g2_grid[1:] + g2_grid[:-1]) * dt
        sigma2_grid[1:] = torch.cumsum(trapezoid, dim=0)

        sigma2_total = sigma2_grid[-1]
        sigma2_bar_grid = sigma2_total - sigma2_grid

        self.t_grid = t_grid
        self.g2_grid = g2_grid
        self.sigma2_grid = sigma2_grid
        self.sigma2_bar_grid = sigma2_bar_grid
        self.sigma1_sq = sigma2_total

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """返回 g(t)^2，支持标量或张量 t（范围 [0,1]）。"""
        beta_0 = self.cfg.beta_0
        beta_1 = self.cfg.beta_1
        return (1.0 - t) * beta_0 + t * beta_1

    def _interp(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """在预计算网格上对任意 t 做线性插值。"""
        t_clamped = t.clamp(0.0, 1.0)
        # 将 t 映射到 [0, n-1]
        n = self.t_grid.numel()
        x = t_clamped * (n - 1)
        idx0 = torch.floor(x).long()
        idx1 = torch.clamp(idx0 + 1, max=n - 1)
        w = x - idx0.to(x.dtype)
        v0 = values[idx0]
        v1 = values[idx1]
        return v0 * (1.0 - w) + v1 * w

    def sigma2(self, t: torch.Tensor) -> torch.Tensor:
        """返回 σ_t^2（与 t 同形状）。"""
        return self._interp(self.sigma2_grid, t.to(self.device, self.dtype))

    def sigma2_bar(self, t: torch.Tensor) -> torch.Tensor:
        """返回 σ̄_t^2（与 t 同形状）。"""
        return self._interp(self.sigma2_bar_grid, t.to(self.device, self.dtype))


