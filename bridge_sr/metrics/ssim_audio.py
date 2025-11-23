from __future__ import annotations

import torch


def _log_spectrogram(
    x: torch.Tensor,
    n_fft: int = 512,
    hop: int = 128,
    win: int = 512,
) -> torch.Tensor:
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
    mag = X.abs()
    log_mag = torch.log1p(mag)
    return log_mag  # (B, F, T_frames)


def compute_ssim_spectrogram(
    x_ref: torch.Tensor,
    x_hat: torch.Tensor,
    n_fft: int = 512,
    hop: int = 128,
    win: int = 512,
) -> float:
    """在对数幅度谱上计算全局 SSIM（简单实现）。"""
    if x_ref.dim() != 1 or x_hat.dim() != 1:
        raise ValueError("x_ref and x_hat must be 1D tensors")
    length = min(x_ref.size(0), x_hat.size(0))
    x_ref = x_ref[:length].to(torch.float32)
    x_hat = x_hat[:length].to(torch.float32)

    S_ref = _log_spectrogram(x_ref, n_fft, hop, win)[0]  # (F, T)
    S_hat = _log_spectrogram(x_hat, n_fft, hop, win)[0]

    x = S_ref.flatten()
    y = S_hat.flatten()

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    C1 = 1e-4
    C2 = 1e-4

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return float(ssim.item())


