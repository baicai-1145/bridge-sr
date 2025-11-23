from __future__ import annotations

from typing import Literal

import torch


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError("x must have shape (T,) or (B, T)")
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = X.abs()  # (B, F, T_frames)
    return mag


def compute_lsd(
    x_ref: torch.Tensor,
    x_hat: torch.Tensor,
    sr: int,
    band: Literal["full", "lf", "hf"] = "full",
    n_fft: int = 2048,
    hop: int = 512,
    win: int = 2048,
) -> float:
    """计算 Log-Spectral Distance (LSD)。

    这里实现的是常见定义：对每一帧在频率维度上计算 dB 差的均方根，再在时间上取平均。
    band:
      - 'full': 全频带
      - 'lf': 低频（<= 4kHz）
      - 'hf': 高频（> 4kHz）
    """
    if x_ref.dim() != 1 or x_hat.dim() != 1:
        raise ValueError("x_ref and x_hat must be 1D tensors")
    length = min(x_ref.size(0), x_hat.size(0))
    x_ref = x_ref[:length].to(torch.float32)
    x_hat = x_hat[:length].to(torch.float32)

    # 归一化到相似幅度范围，避免整体增益差主导 LSD
    for x in (x_ref, x_hat):
        max_val = x.abs().max()
        if max_val > 0:
            x /= max_val

    ref_mag = _stft_mag(x_ref, n_fft, hop, win)[0]  # (F, T)
    hat_mag = _stft_mag(x_hat, n_fft, hop, win)[0]

    eps = 1e-8
    ref_db = 20 * torch.log10(ref_mag + eps)
    hat_db = 20 * torch.log10(hat_mag + eps)

    F, _ = ref_db.shape
    freqs = torch.linspace(0, sr / 2, F, device=ref_db.device)
    if band == "lf":
        mask = freqs <= 4000.0
    elif band == "hf":
        mask = freqs > 4000.0
    else:
        mask = torch.ones_like(freqs, dtype=torch.bool)

    ref_sel = ref_db[mask]
    hat_sel = hat_db[mask]

    if ref_sel.numel() == 0:
        return 0.0

    diff = hat_sel - ref_sel
    lsd_per_frame = torch.sqrt(torch.mean(diff ** 2, dim=0))  # (T_frames,)
    lsd = torch.mean(lsd_per_frame).item()
    return float(lsd)


