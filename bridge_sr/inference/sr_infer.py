from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
import yaml

from bridge_sr.data.lr_generator import LRGenerator  # 仅用于保持一致处理方式
from bridge_sr.models.nuwave2_backbone import BridgeSRBackbone
from bridge_sr.bridge.schedule import BridgeGMaxConfig, BridgeGMaxSchedule
from bridge_sr.bridge.sampler import PFOdeSampler


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def load_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


def chunk_audio(
    x: torch.Tensor,
    segment_length: int,
) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    num_samples = x.size(0)
    if num_samples <= segment_length:
        pad = segment_length - num_samples
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        chunks.append(x)
        return chunks

    start = 0
    while start < num_samples:
        end = start + segment_length
        chunk = x[start:end]
        if chunk.size(0) < segment_length:
            pad = segment_length - chunk.size(0)
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        chunks.append(chunk)
        start = end
    return chunks


def overlap_add(chunks: List[torch.Tensor], original_length: int) -> torch.Tensor:
    # 目前使用简单拼接，最后一块按原始长度裁剪
    x = torch.cat(chunks, dim=0)
    return x[:original_length]


def infer_file(
    model: BridgeSRBackbone,
    sampler: PFOdeSampler,
    audio_path: Path,
    out_path: Path,
    segment_length: int,
    device: torch.device,
) -> None:
    cfg_sr = 48000
    x_lr_48k, sr = load_audio(audio_path, target_sr=cfg_sr)
    original_len = x_lr_48k.size(0)

    chunks = chunk_audio(x_lr_48k, segment_length=segment_length)
    enhanced_chunks: List[torch.Tensor] = []

    for chunk in chunks:
        # 与训练中的 Dataset 一致：幅度归一化到 [-1, 1]
        max_val = chunk.abs().max()
        if max_val > 0:
            norm_chunk = chunk / max_val
        else:
            norm_chunk = chunk

        norm_chunk = norm_chunk.to(device)

        # 使用 PFOdeSampler 从 LR 波形生成 HR 波形
        x_hat = sampler.sample(norm_chunk.unsqueeze(0), steps=50)  # (1, T)
        x_hat = x_hat.squeeze(0)

        # 还原幅度
        if max_val > 0:
            x_hat = x_hat * max_val

        enhanced_chunks.append(x_hat.cpu())

    enhanced = overlap_add(enhanced_chunks, original_length=original_len)
    enhanced = enhanced.clamp(-1.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), enhanced.unsqueeze(0), cfg_sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge-SR inference (any-to-48kHz)")
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
        help="Path to trained checkpoint (.pt) from Stage 1 or finetuned.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory containing wav/flac files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for enhanced 48kHz wav files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of ODE sampling steps.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bridge_cfg = cfg["bridge"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = BridgeSRBackbone(
        channels=int(model_cfg["channels"]),
        resblocks=int(model_cfg["resblocks"]),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule_cfg = BridgeGMaxConfig(
        beta_0=float(bridge_cfg["beta_0"]),
        beta_1=float(bridge_cfg["beta_1"]),
        t_min=float(bridge_cfg.get("t_min", 1.0e-5)),
        t_max=float(bridge_cfg.get("t_max", 1.0)),
        num_grid_points=1000,
    )
    schedule = BridgeGMaxSchedule(schedule_cfg, device=device)
    sampler = PFOdeSampler(
        schedule=schedule,
        model=model,
        scaling=float(bridge_cfg["scaling_factor"]),
        t_min=float(bridge_cfg.get("t_min", 1.0e-5)),
        device=device,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    segment_length = int(data_cfg["segment_length"])

    if input_path.is_file():
        out_file = output_dir / (input_path.stem + "_bridge_sr.wav")
        infer_file(
            model=model,
            sampler=sampler,
            audio_path=input_path,
            out_path=out_file,
            segment_length=segment_length,
            device=device,
        )
    else:
        for ext in ("*.wav", "*.flac"):
            for audio_file in input_path.rglob(ext):
                rel = audio_file.relative_to(input_path)
                out_file = output_dir / rel.with_suffix("_bridge_sr.wav")
                infer_file(
                    model=model,
                    sampler=sampler,
                    audio_path=audio_file,
                    out_path=out_file,
                    segment_length=segment_length,
                    device=device,
                )


if __name__ == "__main__":
    main()

