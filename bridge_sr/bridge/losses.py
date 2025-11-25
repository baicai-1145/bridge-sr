from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn


def bridge_loss(x_hat_0: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
    """Bridge-SR 主损失：预测 x_0 的均方误差。"""
    return torch.mean((x_hat_0 - x_0) ** 2)


STFT_CONFIGS: Sequence[Tuple[int, int, int]] = (
    # (n_fft, hop_length, win_length)
    (1024, 256, 1024),
    (2048, 512, 2048),
    (512, 128, 512),
)


def _compute_stft(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int):
    if x.dim() == 2:
        # (B, T)
        pass
    elif x.dim() == 3 and x.size(1) == 1:
        x = x[:, 0, :]
    else:
        raise ValueError("x must have shape (B, T) or (B, 1, T)")

    # 为了兼容 AMP 下的 fp16 / bf16，这里统一使用 float32 进行 STFT，
    # 避免 cuFFT 不支持低精度类型导致的错误。
    x = x.to(torch.float32)
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    return X


def multi_scale_stft_mag_loss(
    x_hat_0: torch.Tensor,
    x_0: torch.Tensor,
    configs: Sequence[Tuple[int, int, int]] = STFT_CONFIGS,
) -> torch.Tensor:
    """多尺度 STFT 幅度损失。

    对每个尺度计算 |X_hat| 与 |X| 的 L1 距离，然后在尺度上取平均。
    """
    loss = 0.0
    for n_fft, hop_length, win_length in configs:
        X_hat = _compute_stft(x_hat_0, n_fft, hop_length, win_length)
        X = _compute_stft(x_0, n_fft, hop_length, win_length)
        mag_hat = X_hat.abs()
        mag = X.abs()
        loss = loss + torch.mean(torch.abs(mag_hat - mag))
    loss = loss / float(len(configs))
    return loss


def multi_scale_phase_loss(
    x_hat_0: torch.Tensor,
    x_0: torch.Tensor,
    configs: Sequence[Tuple[int, int, int]] = STFT_CONFIGS,
) -> torch.Tensor:
    """多尺度 anti-wrapping phase 损失（近似实现）。

    对各尺度 STFT 的相位差做 wrap 到 [-pi, pi]，然后取 L1 平均。
    """
    loss = 0.0
    for n_fft, hop_length, win_length in configs:
        X_hat = _compute_stft(x_hat_0, n_fft, hop_length, win_length)
        X = _compute_stft(x_0, n_fft, hop_length, win_length)
        phase_hat = torch.angle(X_hat)
        phase = torch.angle(X)
        dphase = phase_hat - phase
        # wrap 到 [-pi, pi]
        dphase_wrapped = torch.atan2(torch.sin(dphase), torch.cos(dphase))
        loss = loss + torch.mean(torch.abs(dphase_wrapped))
    loss = loss / float(len(configs))
    return loss

