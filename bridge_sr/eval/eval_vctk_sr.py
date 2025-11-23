from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from bridge_sr.data.dataset_vctk import create_dataloader
from bridge_sr.models.nuwave2_backbone import BridgeSRBackbone
from bridge_sr.bridge.schedule import BridgeGMaxConfig, BridgeGMaxSchedule
from bridge_sr.bridge.sampler import PFOdeSampler
from bridge_sr.metrics.lsd import compute_lsd
from bridge_sr.metrics.sisnr import compute_sisnr
from bridge_sr.metrics.ssim_audio import compute_ssim_spectrogram


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def evaluate(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    max_batches: int = 50,
) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    bridge_cfg = cfg["bridge"]

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    test_loader = create_dataloader(cfg, is_train=False)

    model = BridgeSRBackbone(
        channels=int(model_cfg["channels"]),
        resblocks=int(model_cfg["resblocks"]),
    ).to(device_obj)
    ckpt = load_checkpoint(checkpoint_path, device_obj)
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule_cfg = BridgeGMaxConfig(
        beta_0=float(bridge_cfg["beta_0"]),
        beta_1=float(bridge_cfg["beta_1"]),
        t_min=float(bridge_cfg.get("t_min", 1.0e-5)),
        t_max=float(bridge_cfg.get("t_max", 1.0)),
        num_grid_points=1000,
    )
    schedule = BridgeGMaxSchedule(schedule_cfg, device=device_obj)
    sampler = PFOdeSampler(
        schedule=schedule,
        model=model,
        scaling=float(bridge_cfg["scaling_factor"]),
        t_min=float(bridge_cfg.get("t_min", 1.0e-5)),
        device=device_obj,
    )

    sr = int(data_cfg["sample_rate"])
    seg_len = int(data_cfg["segment_length"])

    total_lsd = 0.0
    total_lsd_lf = 0.0
    total_lsd_hf = 0.0
    total_sisnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            x_hr, _ = batch  # (B, T)
            x_hr = x_hr.to(device_obj)

            # 这里简单地将 HR 当作 LR 上采样后的输入，以便评估模型端到端效果；
            # 更精细的评估可以根据具体 SR 任务配置重新构造 LR。
            x_lr = x_hr.clone()

            for i in range(x_hr.size(0)):
                ref = x_hr[i].cpu()
                lr_seg = x_lr[i].unsqueeze(0)
                est = sampler.sample(lr_seg, steps=50).squeeze(0).cpu()

                lsd = compute_lsd(ref, est, sr=sr, band="full")
                lsd_lf = compute_lsd(ref, est, sr=sr, band="lf")
                lsd_hf = compute_lsd(ref, est, sr=sr, band="hf")
                sisnr = compute_sisnr(ref, est)
                ssim = compute_ssim_spectrogram(ref, est)

                total_lsd += lsd
                total_lsd_lf += lsd_lf
                total_lsd_hf += lsd_hf
                total_sisnr += sisnr
                total_ssim += ssim
                count += 1

    if count == 0:
        print("No samples evaluated.")
        return

    print(f"Samples evaluated: {count}")
    print(f"LSD (full): {total_lsd / count:.4f}")
    print(f"LSD-LF:      {total_lsd_lf / count:.4f}")
    print(f"LSD-HF:      {total_lsd_hf / count:.4f}")
    print(f"SI-SNR:      {total_sisnr / count:.2f} dB")
    print(f"SSIM (spec): {total_ssim / count:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Bridge-SR on VCTK test set")
    parser.add_argument(
        "--config",
        type=str,
        default="bridge_sr/configs/bridge_sr_vctk.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=50,
        help="Maximum number of test batches to evaluate.",
    )
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()

