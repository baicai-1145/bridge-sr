from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import yaml

from bridge_sr.utils import seed_everything
from bridge_sr.data.dataset_vctk import create_dataloader
from bridge_sr.data.lr_generator import LRGenerator
from bridge_sr.models.nuwave2_backbone import BridgeSRBackbone
from bridge_sr.bridge.schedule import BridgeGMaxConfig, BridgeGMaxSchedule
from bridge_sr.bridge.sampler import BridgeForwardSampler
from bridge_sr.bridge.losses import bridge_loss


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


@torch.no_grad()
def evaluate_bridge_loss(
    model: nn.Module,
    sampler: BridgeForwardSampler,
    dataloader,
    lr_generator: LRGenerator,
    scaling: float,
    device: torch.device,
    max_batches: int = 10,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x_hr, _ = batch
        x_hr = x_hr.to(device)

        x_lr_list = []
        for i in range(x_hr.size(0)):
            x_lr_i, _ = lr_generator(x_hr[i])
            x_lr_list.append(x_lr_i)
        x_lr = torch.stack(x_lr_list, dim=0).to(device)

        x0 = scaling * x_hr
        xT = scaling * x_lr

        t = torch.rand(x0.size(0), device=device)
        x_t = sampler.sample(x0, xT, t)

        x_hat0 = model(x_t, t, xT)
        loss = bridge_loss(x_hat0, x0)

        total_loss += loss.item()
        total_batches += 1

    model.train()
    if total_batches == 0:
        return 0.0
    return total_loss / total_batches


def train_bridge(
    config_path: str,
    device: str = "cuda",
    max_steps: int | None = None,
) -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    bridge_cfg = cfg["bridge"]

    seed_everything(int(train_cfg.get("seed", 42)))

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    dataloader = create_dataloader(cfg, is_train=True)
    val_dataloader = create_dataloader(cfg, is_train=False)
    lr_generator = LRGenerator(
        sr_target=int(cfg["data"]["sample_rate"]),
        sr_min=6000,
        sr_max=48000,
    )

    model = BridgeSRBackbone(
        channels=int(cfg["model"]["channels"]),
        resblocks=int(cfg["model"]["resblocks"]),
    ).to(device_obj)

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

    # 混合精度
    use_amp = device_obj.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # TensorBoard
    log_dir = Path("runs") / "bridge_sr_stage1"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    total_steps = int(train_cfg["total_steps"])
    eval_interval = int(train_cfg.get("eval_interval", 5000))
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    scaling = float(bridge_cfg["scaling_factor"])

    model.train()
    step = 0
    try:
        while step < total_steps:
            for batch in dataloader:
                x_hr, _ = batch  # (B, T), 已是 48kHz
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

                with autocast(device_type=device_obj.type, enabled=use_amp):
                    x_hat0 = model(x_t, t, xT)
                    loss = bridge_loss(x_hat0, x0)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                step += 1

                # TensorBoard 日志
                writer.add_scalar("train/bridge_loss", loss.item(), step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], step
                )

                if step % int(train_cfg.get("log_interval", 100)) == 0:
                    print(f"[step {step}] bridge_loss = {loss.item():.6f}")

                if step % eval_interval == 0:
                    val_loss = evaluate_bridge_loss(
                        model=model,
                        sampler=sampler,
                        dataloader=val_dataloader,
                        lr_generator=lr_generator,
                        scaling=scaling,
                        device=device_obj,
                        max_batches=10,
                    )
                    writer.add_scalar("val/bridge_loss", val_loss, step)
                    print(f"[eval step {step}] val_bridge_loss = {val_loss:.6f}")

                if step % int(train_cfg.get("save_interval", 50000)) == 0:
                    ckpt_dir = Path("checkpoints")
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"bridge_step_{step}.pt"
                    save_checkpoint(str(ckpt_path), model, optimizer, step, cfg)

                if step >= total_steps:
                    break
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bridge-SR (Stage 1: bridge loss only)")
    parser.add_argument(
        "--config",
        type=str,
        default="bridge_sr/configs/bridge_sr_vctk.yaml",
        help="Path to YAML config file.",
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
        help="Optional maximum training steps for debugging.",
    )
    args = parser.parse_args()

    train_bridge(args.config, device=args.device, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
