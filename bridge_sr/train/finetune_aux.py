from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import yaml

from bridge_sr.utils import seed_everything
from bridge_sr.data.dataset_vctk import create_dataloader
from bridge_sr.data.lr_generator import LRGenerator
from bridge_sr.models.nuwave2_backbone import BridgeSRBackbone
from bridge_sr.bridge.schedule import BridgeGMaxConfig, BridgeGMaxSchedule
from bridge_sr.bridge.sampler import BridgeForwardSampler
from bridge_sr.bridge.losses import (
    bridge_loss,
    multi_scale_stft_mag_loss,
    multi_scale_phase_loss,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Adam,
    step: int,
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(ckpt, path)


def finetune_aux(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    max_steps: int | None = None,
) -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    bridge_cfg = cfg["bridge"]
    loss_cfg = cfg["loss"]

    seed_everything(int(train_cfg.get("seed", 42)))

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    dataloader = create_dataloader(cfg, is_train=True)
    lr_generator = LRGenerator(
        sr_target=int(cfg["data"]["sample_rate"]),
        sr_min=6000,
        sr_max=48000,
    )

    model = BridgeSRBackbone(
        channels=int(cfg["model"]["channels"]),
        resblocks=int(cfg["model"]["resblocks"]),
    ).to(device_obj)

    ckpt = load_checkpoint(checkpoint_path, device_obj)
    model.load_state_dict(ckpt["model"])

    schedule_cfg = BridgeGMaxConfig(
        beta_0=float(bridge_cfg["beta_0"]),
        beta_1=float(bridge_cfg["beta_1"]),
        t_min=float(bridge_cfg.get("t_min", 0.0)),
        t_max=float(bridge_cfg.get("t_max", 1.0)),
        num_grid_points=1000,
    )
    schedule = BridgeGMaxSchedule(schedule_cfg, device=device_obj)
    sampler = BridgeForwardSampler(schedule, device=device_obj)

    optimizer = Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass

    total_steps = int(train_cfg.get("finetune_steps", 70000))
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    scaling = float(bridge_cfg["scaling_factor"])
    lambda_mag = float(loss_cfg["lambda_mag"])
    lambda_phase = float(loss_cfg["lambda_phase"])

    # TensorBoard
    log_dir = Path("runs") / "bridge_sr_finetune"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # 精度与 AMP / TF32 配置
    prec = str(train_cfg.get("precision", "fp32")).lower()
    use_amp = False
    amp_dtype = None
    if device_obj.type == "cuda":
        if "tf32" in prec:
            torch.set_float32_matmul_precision("medium")
        else:
            torch.set_float32_matmul_precision("highest")

        if "fp16" in prec:
            use_amp = True
            amp_dtype = torch.float16
        elif "bf16" in prec:
            use_amp = True
            amp_dtype = torch.bfloat16
    if device_obj.type == "cuda":
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler("cpu", enabled=False)

    model.train()
    step = 0
    try:
        while step < total_steps:
            for batch in dataloader:
                x_hr, _ = batch
                x_hr = x_hr.to(device_obj)

                x_lr_list = []
                for i in range(x_hr.size(0)):
                    x_lr_i, _ = lr_generator(x_hr[i])
                    x_lr_list.append(x_lr_i)
                x_lr = torch.stack(x_lr_list, dim=0).to(device_obj)

                x0 = scaling * x_hr
                xT = scaling * x_lr

                t = torch.rand(x0.size(0), device=device_obj)
                x_t = sampler.sample(x0, xT, t)

                ctx = autocast(device_obj.type, dtype=amp_dtype) if use_amp else nullcontext()
                with ctx:
                    x_hat0 = model(x_t, t, xT)

                    # Bridge loss 在缩放空间上计算
                    l_bridge = bridge_loss(x_hat0, x0)

                    # STFT / phase 损失在还原尺度的波形上计算
                    x_hat0_unscaled = x_hat0 / scaling
                    l_mag = multi_scale_stft_mag_loss(x_hat0_unscaled, x_hr)
                    l_phase = multi_scale_phase_loss(x_hat0_unscaled, x_hr)

                    loss = l_bridge + lambda_mag * l_mag + lambda_phase * l_phase

                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                step += 1

                # TensorBoard 日志
                writer.add_scalar("train/bridge_loss", l_bridge.item(), step)
                writer.add_scalar("train/mag_loss", l_mag.item(), step)
                writer.add_scalar("train/phase_loss", l_phase.item(), step)
                writer.add_scalar("train/total_loss", loss.item(), step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], step
                )

                if step % int(train_cfg.get("log_interval", 100)) == 0:
                    print(
                        f"[finetune step {step}] "
                        f"bridge={l_bridge.item():.6f}, "
                        f"mag={l_mag.item():.6f}, "
                        f"phase={l_phase.item():.6f}, "
                        f"total={loss.item():.6f}"
                    )

                if step % int(train_cfg.get("save_interval", 50000)) == 0:
                    ckpt_dir = Path("checkpoints_finetune")
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"bridge_finetune_step_{step}.pt"
                    save_checkpoint(str(ckpt_path), model, optimizer, step, cfg)

                if step >= total_steps:
                    break
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune Bridge-SR with STFT/phase losses (Stage 2)")
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
        help="Path to pretrained Stage-1 checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum finetune steps for debugging.",
    )
    args = parser.parse_args()

    finetune_aux(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
