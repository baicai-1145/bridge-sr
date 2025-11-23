import random
from typing import Tuple

import torch
import torchaudio


class LRGenerator:
    """Generate low-resolution waveform from high-resolution waveform.

    输入为 48kHz 波形，先下采样到随机采样率，再上采样回 48kHz。
    返回的波形长度与输入一致，采样率信息单独返回。
    """

    def __init__(
        self,
        sr_target: int = 48000,
        sr_min: int = 6000,
        sr_max: int = 48000,
    ) -> None:
        assert sr_min > 0 and sr_min <= sr_max <= sr_target
        self.sr_target = sr_target
        self.sr_min = sr_min
        self.sr_max = sr_max

    def _sample_input_sr(self) -> int:
        # 采样一个 1kHz 步长的采样率，覆盖 [sr_min, sr_max]
        k_min = self.sr_min // 1000
        k_max = self.sr_max // 1000
        k = random.randint(k_min, k_max)
        return k * 1000

    def __call__(self, x_hr: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x_hr: 1D or 2D tensor, shape (T,) or (1, T), assumed at sr_target.

        Returns:
            x_lr_48k: 1D tensor at sr_target, degraded then upsampled.
            sr_in: sampled low-resolution sampling rate.
        """
        if x_hr.ndim == 1:
            wav = x_hr.unsqueeze(0)
        elif x_hr.ndim == 2 and x_hr.size(0) == 1:
            wav = x_hr
        else:
            raise ValueError("x_hr must be 1D or shape (1, T)")

        orig_len = wav.size(-1)

        sr_in = self._sample_input_sr()
        if sr_in == self.sr_target:
            return wav.squeeze(0), sr_in

        # 下采样
        wav_lr = torchaudio.functional.resample(
            wav,
            orig_freq=self.sr_target,
            new_freq=sr_in,
        )
        # 再上采样回目标采样率
        wav_lr_48k = torchaudio.functional.resample(
            wav_lr,
            orig_freq=sr_in,
            new_freq=self.sr_target,
        )

        # 对齐长度：裁剪或在末尾补零，保证输出与输入长度一致
        out_len = wav_lr_48k.size(-1)
        if out_len > orig_len:
            wav_lr_48k = wav_lr_48k[..., :orig_len]
        elif out_len < orig_len:
            pad = orig_len - out_len
            wav_lr_48k = torch.nn.functional.pad(wav_lr_48k, (0, pad))

        return wav_lr_48k.squeeze(0), sr_in


def _impulse_alignment_test(generator: LRGenerator, length: int = 32768) -> int:
    """简单的脉冲对齐测试工具函数。

    返回下采样-上采样前后主峰位置之差（单位：采样点）。
    该函数仅用于手工调试，不在训练脚本中调用。
    """
    x = torch.zeros(length)
    center = length // 2
    x[center] = 1.0
    x_lr, _ = generator(x)
    peak_before = center
    peak_after = int(torch.argmax(x_lr).item())
    return peak_after - peak_before


def _sine_highfreq_attenuation_test() -> Tuple[float, float]:
    """使用高频成分的正弦波测试 LR 生成是否衰减高频。

    返回 (原始高频能量, LR 高频能量)，便于观察。
    """
    sr_target = 48000
    length = 32768
    t = torch.arange(length, dtype=torch.float32) / sr_target

    # 低频 1kHz + 高频 10kHz
    x_hr = 0.5 * torch.sin(2 * torch.pi * 1000 * t) + 0.5 * torch.sin(
        2 * torch.pi * 10000 * t
    )

    gen = LRGenerator(sr_target=sr_target, sr_min=8000, sr_max=8000)
    x_lr, _ = gen(x_hr)

    assert x_lr.shape == x_hr.shape

    X_hr = torch.fft.rfft(x_hr)
    X_lr = torch.fft.rfft(x_lr)

    freqs = torch.linspace(0, sr_target / 2, X_hr.numel())
    high_mask = freqs > 4000.0

    hr_high = (X_hr[high_mask].abs() ** 2).mean().item()
    lr_high = (X_lr[high_mask].abs() ** 2).mean().item()
    return hr_high, lr_high


if __name__ == "__main__":
    gen = LRGenerator()
    shift = _impulse_alignment_test(gen)
    print(f"Impulse peak shift (samples): {shift}")

    hr_high, lr_high = _sine_highfreq_attenuation_test()
    print(f"High-band energy before: {hr_high:.6e}, after: {lr_high:.6e}")
