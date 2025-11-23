from __future__ import annotations

import torch


def compute_sisnr(
    x_ref: torch.Tensor,
    x_hat: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """计算 SI-SNR（Scale-Invariant Signal-to-Noise Ratio）。"""
    if x_ref.dim() != 1 or x_hat.dim() != 1:
        raise ValueError("x_ref and x_hat must be 1D tensors")
    length = min(x_ref.size(0), x_hat.size(0))
    s = x_ref[:length].to(torch.float32)
    s_hat = x_hat[:length].to(torch.float32)

    # 去除直流分量
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()

    # 投影
    s_energy = torch.sum(s ** 2) + eps
    proj = torch.sum(s_hat * s) * s / s_energy
    noise = s_hat - proj

    ratio = (torch.sum(proj ** 2) + eps) / (torch.sum(noise ** 2) + eps)
    sisnr = 10 * torch.log10(ratio)
    return float(sisnr.item())


