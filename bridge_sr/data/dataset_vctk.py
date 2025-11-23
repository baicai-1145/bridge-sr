import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio


@dataclass
class VCTKDatasetConfig:
    root: str
    wav_root: str
    sample_rate: int
    segment_length: int
    train_speakers: Sequence[str]
    test_speakers: Sequence[str]
    batch_size: int
    num_workers: int
    cache_train_in_memory: bool = False
    cache_test_in_memory: bool = False


def _list_speakers(wav_root: str) -> List[str]:
    speakers: List[str] = []
    for name in os.listdir(wav_root):
        full = os.path.join(wav_root, name)
        if os.path.isdir(full):
            speakers.append(name)
    speakers.sort()
    return speakers


def _collect_files(
    wav_root: str,
    speakers: Iterable[str],
    segment_length: int,
    sample_rate: int,
) -> List[str]:
    """Collect utterance file paths with at least segment_length samples."""
    paths: List[str] = []
    for spk in speakers:
        spk_dir = os.path.join(wav_root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for fname in os.listdir(spk_dir):
            if not (fname.endswith(".flac") or fname.endswith(".wav")):
                continue
            path = os.path.join(spk_dir, fname)
            try:
                info = torchaudio.info(path)
            except Exception:
                continue
            num_frames = info.num_frames
            sr = info.sample_rate
            if sr != sample_rate:
                # 估算重采样后的帧数，避免加载音频做长度判断
                num_frames = int(num_frames * float(sample_rate) / float(sr) + 0.5)
            if num_frames >= segment_length:
                paths.append(path)
    paths.sort()
    return paths


class VCTKWaveformDataset(Dataset):
    def __init__(
        self,
        cfg: VCTKDatasetConfig,
        speakers: Sequence[str],
        is_train: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.wav_root = cfg.wav_root
        self.sample_rate = cfg.sample_rate
        self.segment_length = cfg.segment_length
        self.is_train = is_train
        self.normalize = normalize

        self.paths: List[str] = _collect_files(
            self.wav_root,
            speakers,
            self.segment_length,
            self.sample_rate,
        )
        if not self.paths:
            raise RuntimeError(
                f"No audio files found for speakers {speakers} under {self.wav_root}"
            )

        # 可选：预先将所有波形加载到内存，减少每 step 的磁盘 I/O
        self.cache_in_memory = (
            cfg.cache_train_in_memory if is_train else cfg.cache_test_in_memory
        )
        self._wave_cache: Optional[List[torch.Tensor]] = None
        if self.cache_in_memory:
            self._wave_cache = []
            for path in self.paths:
                wav, sr = torchaudio.load(path)
                if wav.ndim == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                wav = wav.squeeze(0)
                self._wave_cache.append(wav)

    def _load_wave(self, index: int) -> torch.Tensor:
        if self.cache_in_memory and self._wave_cache is not None:
            return self._wave_cache[index]

        path = self.paths[index]
        wav, sr = torchaudio.load(path)
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[index]
        wav = self._load_wave(index)

        num_samples = wav.size(0)
        if num_samples == self.segment_length:
            start = 0
        elif num_samples > self.segment_length:
            max_start = num_samples - self.segment_length
            start = random.randint(0, max_start)
        else:
            # 此情形理论上已在文件收集阶段过滤，这里退而求其次做末尾零填充
            pad = self.segment_length - num_samples
            wav = torch.nn.functional.pad(wav, (0, pad))
            start = 0

        seg = wav[start : start + self.segment_length]

        if self.normalize:
            max_val = seg.abs().max()
            if max_val > 0:
                seg = seg / max_val

        # 也返回说话人 ID，便于后续分析
        spk = os.path.basename(os.path.dirname(path))
        return seg, spk


def build_vctk_config_from_yaml_dict(cfg_dict: Dict) -> VCTKDatasetConfig:
    data_cfg = cfg_dict["data"]
    train_cfg = cfg_dict["train"]
    return VCTKDatasetConfig(
        root=data_cfg["root"],
        wav_root=data_cfg["wav_root"],
        sample_rate=int(data_cfg["sample_rate"]),
        segment_length=int(data_cfg["segment_length"]),
        train_speakers=list(data_cfg.get("train_speakers", [])),
        test_speakers=list(data_cfg.get("test_speakers", [])),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        cache_train_in_memory=bool(data_cfg.get("cache_train_in_memory", False)),
        cache_test_in_memory=bool(data_cfg.get("cache_test_in_memory", False)),
    )


def _select_speakers(
    cfg: VCTKDatasetConfig,
    wav_root: str,
    is_train: bool,
) -> List[str]:
    all_speakers = _list_speakers(wav_root)
    test_set = set(cfg.test_speakers)

    if is_train:
        if cfg.train_speakers:
            selected = [s for s in cfg.train_speakers if s in all_speakers]
        else:
            selected = [s for s in all_speakers if s not in test_set]
    else:
        selected = [s for s in all_speakers if s in test_set]

    selected.sort()
    if not selected:
        raise RuntimeError(
            f"No {'train' if is_train else 'test'} speakers selected under {wav_root}"
        )
    return selected


def create_dataloader(
    cfg_dict: Dict,
    is_train: bool = True,
) -> DataLoader:
    cfg = build_vctk_config_from_yaml_dict(cfg_dict)
    speakers = _select_speakers(cfg, cfg.wav_root, is_train=is_train)
    dataset = VCTKWaveformDataset(cfg, speakers=speakers, is_train=is_train)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=is_train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader
