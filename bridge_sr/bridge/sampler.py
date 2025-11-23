from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from .schedule import BridgeGMaxSchedule


class BridgeForwardSampler:
    """用于训练阶段的前向桥采样器：从 (x_0, x_T, t) 生成 x_t。

    在 Bridge-SR 设定下：
      - f(t) = 0, α_t = 1
      - μ_t = (σ̄_t^2 / σ_1^2) x_0 + (σ_t^2 / σ_1^2) x_T
      - Σ_t = (σ_t^2 σ̄_t^2 / σ_1^2) I
    """

    def __init__(
        self,
        schedule: BridgeGMaxSchedule,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.schedule = schedule
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    def sample(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """从桥分布采样 x_t。

        Args:
            x0: 高分辨率目标波形，形状 (B, T) 或 (B, 1, T)
            xT: 低分辨率波形，形状与 x0 相同
            t: 采样时间，形状 (B,) 或标量张量，取值范围 [0,1]
            noise: 可选外部噪声张量，与 x0 形状相同；若为 None，则内部采样 N(0, I)

        Returns:
            x_t: 与 x0 形状一致的带噪中间状态。
        """
        if x0.shape != xT.shape:
            raise ValueError("x0 and xT must have the same shape")

        x0 = x0.to(self.device, self.dtype)
        xT = xT.to(self.device, self.dtype)

        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.to(self.device, self.dtype)
        if t.numel() == 1 and x0.size(0) > 1:
            t = t.expand(x0.size(0))

        sigma2 = self.schedule.sigma2(t)           # (B,)
        sigma2_bar = self.schedule.sigma2_bar(t)   # (B,)
        sigma1_sq = self.schedule.sigma1_sq        # 标量

        # 调整形状以便广播到 (B, T) 或 (B, 1, T)
        while sigma2.ndim < x0.ndim:
            sigma2 = sigma2.unsqueeze(-1)
            sigma2_bar = sigma2_bar.unsqueeze(-1)

        coef0 = sigma2_bar / sigma1_sq
        coefT = sigma2 / sigma1_sq

        mu = coef0 * x0 + coefT * xT
        var = (sigma2 * sigma2_bar) / sigma1_sq
        std = torch.sqrt(torch.clamp(var, min=1e-12))

        if noise is None:
            noise = torch.randn_like(mu)
        else:
            noise = noise.to(self.device, self.dtype)

        x_t = mu + std * noise
        return x_t


class PFOdeSampler:
    """简化版概率流 ODE 采样器，用于推理阶段从 x_T 生成 x_0。

    使用网络在一系列时间步上迭代更新 x_t：
      1) 给定当前 x_t 和 t，预测 x_0_hat = model(x_t, t, x_T)
      2) 使用桥的解析公式在下一时间步 t_next 上计算均值 μ_{t_next}(x_0_hat, x_T)
      3) 将 x_{t_next} 设为该均值（忽略扩散项），重复直到最小时间步
      4) 返回最后一次预测的 x_0_hat 作为生成的高分辨率波形
    """

    def __init__(
        self,
        schedule: BridgeGMaxSchedule,
        model: torch.nn.Module,
        scaling: float,
        t_min: float = 1.0e-5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.schedule = schedule
        self.model = model
        self.scaling = float(scaling)
        self.t_min = float(t_min)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    @torch.no_grad()
    def sample(
        self,
        x_lr: torch.Tensor,
        steps: int = 50,
    ) -> torch.Tensor:
        """给定低分辨率波形生成高分辨率波形。

        Args:
            x_lr: (B, T) 低分辨率波形（已上采样到目标采样率）
            steps: 采样步数（例如 50, 8, 4 等）

        Returns:
            x_hat_0: (B, T) 生成的高分辨率波形（未缩放，需由调用方决定是否再做归一化）
        """
        if x_lr.dim() == 1:
            x_lr = x_lr.unsqueeze(0)
        if x_lr.dim() != 2:
            raise ValueError("x_lr must have shape (B, T) or (T,)")

        x_lr = x_lr.to(self.device, self.dtype)

        # 缩放到训练使用的尺度
        xT = self.scaling * x_lr
        x_t = xT.clone()

        t_grid = torch.linspace(
            1.0,
            self.t_min,
            steps,
            device=self.device,
            dtype=self.dtype,
        )

        x0_hat_last = None
        for i in range(steps):
            t = t_grid[i]
            t_batch = torch.full(
                (x_t.size(0),),
                t.item(),
                device=self.device,
                dtype=self.dtype,
            )

            # 模型预测当前时间步的 x_0
            x0_hat = self.model(x_t, t_batch, xT)
            x0_hat_last = x0_hat

            # 最后一步只需要 x0_hat，不再更新 x_t
            if i == steps - 1:
                break

            t_next = t_grid[i + 1]
            sigma2_next = self.schedule.sigma2(t_next)
            sigma2_bar_next = self.schedule.sigma2_bar(t_next)
            sigma1_sq = self.schedule.sigma1_sq

            # 调整形状以广播
            while sigma2_next.ndim < x0_hat.ndim:
                sigma2_next = sigma2_next.unsqueeze(-1)
                sigma2_bar_next = sigma2_bar_next.unsqueeze(-1)

            coef0 = sigma2_bar_next / sigma1_sq
            coefT = sigma2_next / sigma1_sq

            # 使用预测的 x0_hat 和固定的 xT 计算下一时间步的均值
            x_t = coef0 * x0_hat + coefT * xT

        if x0_hat_last is None:
            raise RuntimeError("Sampling did not run any steps.")

        # 反缩放回原始波形尺度
        x_hat_0 = x0_hat_last / self.scaling
        return x_hat_0


