"""
Dataset utilities for conditional AMT noise diffusion.
扫描 Real-Len*/Real-2048/Synthetic-Norm
    自动短噪声嵌入到 2048
    返回 noise/type_id/length_ratio/source_id

Expected real layout:
    noise_data/Real-Len100\\Pulse\\*.npy
    ...
    noise_data/Real-2048\\Triangular-wave\\*.npy

Expected synthetic layout:
    noise_data/Synthetic-Norm\\Synthetic-Len100\\Pulse\\*.npy
    ...
    noise_data/Synthetic-Norm\\Synthetic-Len2048\\Composite\\*.npy

把实测噪声段和合成噪声段统一整理成固定长度 1×2048 的单通道噪声样本，并为每个样本附带噪声类型、有效长度比例和数据来源标签，用于条件扩散模型训练。

实测噪声 npy / 合成噪声 npy
        ↓
扫描目录，识别 source、type、length
        ↓
封装成 NoiseItem
        ↓
训练时读取 npy
        ↓
RMS / maxabs 归一化
        ↓
裁剪或零填充到 2048
        ↓
随机位置嵌入短噪声
        ↓
时间翻转 / 符号翻转
        ↓
返回：
noise:        (1, 2048)
type_id:      噪声类别
length_ratio: 有效长度比例
source_id:    real / synthetic
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


NOISE_TYPE_NAMES = [
    "Pulse",
    "Quasi-Sinusoidal",
    "Square-wave",
    "Step-wave",
    "Triangular-wave",
    "Power-frequency",
    "White-noise",
    "Color-noise",
    "Harmonic",
    "Composite",
]
TYPE_TO_ID = {name: idx for idx, name in enumerate(NOISE_TYPE_NAMES)}
SOURCE_TO_ID = {"real": 0, "synthetic": 1}

LEN_RE = re.compile(r"_len(\d+)")
REAL_LEN_DIR_RE = re.compile(r"^Real-Len(\d+)$")
SYN_LEN_DIR_RE = re.compile(r"^Synthetic-Len(\d+)$")


@dataclass(frozen=True)
class NoiseItem:
    path: Path
    type_name: str
    type_id: int
    source_name: str
    source_id: int
    original_len: int


def parse_len_from_name(path: Path) -> int | None:
    match = LEN_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def parse_len_from_dir(path: Path, prefix: str) -> int | None:
    pattern = REAL_LEN_DIR_RE if prefix == "Real" else SYN_LEN_DIR_RE
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def normalize_1d(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "none":
        return x
    if mode == "rms":
        x = x - float(np.mean(x))
        rms = float(np.sqrt(np.mean(x * x)))
        if rms > 1e-8:
            x = x / rms
        return x.astype(np.float32, copy=False)
    if mode == "maxabs":
        max_abs = float(np.max(np.abs(x)))
        if max_abs > 1e-8:
            x = x / max_abs
        return x.astype(np.float32, copy=False)
    raise ValueError(f"Unknown normalize mode: {mode}")


class NoiseNPYDataset(Dataset):
    """Load fixed-size clips from all supported lengths or selected original lengths."""

    def __init__(
        self,
        data_root=r"noise_data",
        synthetic_root=None,
        target_length=2048,
        include_real=True,
        include_synthetic=True,
        max_real_original_len=2048,
        normalize="rms",
        random_embed=True,
        flip_prob=0.5,
        sign_flip_prob=0.5,
        return_metadata=False,
        selected_lengths=None,
    ):
        self.data_root = Path(data_root)
        self.synthetic_root = (
            Path(synthetic_root)
            if synthetic_root is not None
            else self.data_root / "Synthetic-Norm"
        )
        self.target_length = int(target_length)
        self.selected_lengths = None
        if selected_lengths is not None:
            selected_lengths = tuple(selected_lengths)
            if not selected_lengths or any(
                isinstance(length, bool)
                or not isinstance(length, (int, np.integer))
                or not 0 < length <= self.target_length
                for length in selected_lengths
            ):
                raise ValueError(
                    "selected_lengths must be a non-empty collection of positive "
                    f"integers no larger than target_length ({self.target_length})"
                )
            self.selected_lengths = frozenset(int(length) for length in selected_lengths)
        self.normalize = normalize
        self.random_embed = random_embed
        self.flip_prob = float(flip_prob)
        self.sign_flip_prob = float(sign_flip_prob)
        self.return_metadata = return_metadata

        items: list[NoiseItem] = []
        if include_real:
            items.extend(self._scan_real(max_real_original_len=max_real_original_len))
        if include_synthetic:
            items.extend(self._scan_synthetic())
        if not items:
            raise FileNotFoundError(
                f"No noise npy files found. data_root={self.data_root}, "
                f"synthetic_root={self.synthetic_root}"
            )

        self.items = items

    def _scan_real(self, max_real_original_len: int) -> list[NoiseItem]:
        items: list[NoiseItem] = []
        if not self.data_root.exists():
            return items

        for len_dir in sorted(self.data_root.iterdir()):
            if not len_dir.is_dir():
                continue

            if len_dir.name == "Real-2048":
                dir_len = 2048
            else:
                dir_match = REAL_LEN_DIR_RE.match(len_dir.name)
                if dir_match is None:
                    continue
                dir_len = int(dir_match.group(1))

            if max_real_original_len is not None and dir_len > max_real_original_len:
                continue

            for npy_path in sorted(len_dir.glob("*/*.npy")):
                type_name = npy_path.parent.name
                if type_name not in TYPE_TO_ID:
                    continue
                original_len = parse_len_from_name(npy_path) or dir_len
                if max_real_original_len is not None and original_len > max_real_original_len:
                    continue
                if self.selected_lengths is not None and original_len not in self.selected_lengths:
                    continue
                items.append(
                    NoiseItem(
                        path=npy_path,
                        type_name=type_name,
                        type_id=TYPE_TO_ID[type_name],
                        source_name="real",
                        source_id=SOURCE_TO_ID["real"],
                        original_len=min(int(original_len), self.target_length),
                    )
                )
        return items

    def _scan_synthetic(self) -> list[NoiseItem]:
        items: list[NoiseItem] = []
        if not self.synthetic_root.exists():
            return items

        for len_dir in sorted(self.synthetic_root.iterdir()):
            if not len_dir.is_dir():
                continue
            dir_match = SYN_LEN_DIR_RE.match(len_dir.name)
            if dir_match is None:
                continue
            dir_len = int(dir_match.group(1))
            if dir_len > self.target_length:
                continue

            for npy_path in sorted(len_dir.glob("*/*.npy")):
                type_name = npy_path.parent.name
                if type_name not in TYPE_TO_ID:
                    continue
                original_len = parse_len_from_name(npy_path) or dir_len
                if self.selected_lengths is not None and original_len not in self.selected_lengths:
                    continue
                items.append(
                    NoiseItem(
                        path=npy_path,
                        type_name=type_name,
                        type_id=TYPE_TO_ID[type_name],
                        source_name="synthetic",
                        source_id=SOURCE_TO_ID["synthetic"],
                        original_len=min(int(original_len), self.target_length),
                    )
                )
        return items

    def __len__(self):
        return len(self.items)

    def _to_fixed_length(self, x: np.ndarray, original_len: int) -> tuple[np.ndarray, int]:
        n = int(x.shape[0])
        if n >= self.target_length:
            if n == self.target_length:
                return x[: self.target_length], 0
            max_start = n - self.target_length
            start = np.random.randint(0, max_start + 1) if self.random_embed else max_start // 2
            return x[start : start + self.target_length], int(start)

        out = np.zeros(self.target_length, dtype=np.float32)
        max_start = self.target_length - n
        start = np.random.randint(0, max_start + 1) if self.random_embed else max_start // 2
        out[start : start + n] = x

        return out, int(start)

    def __getitem__(self, idx):
        item = self.items[int(idx)]
        x = np.load(item.path)
        x = normalize_1d(x, self.normalize)
        x, offset = self._to_fixed_length(x, item.original_len)

        if self.flip_prob > 0 and torch.rand(1).item() < self.flip_prob:
            x = x[::-1].copy()
        if self.sign_flip_prob > 0 and torch.rand(1).item() < self.sign_flip_prob:
            x = -x

        noise = torch.from_numpy(np.ascontiguousarray(x[None, :])).float()
        type_id = torch.tensor(item.type_id, dtype=torch.long)
        source_id = torch.tensor(item.source_id, dtype=torch.long)
        length = torch.tensor(item.original_len, dtype=torch.long)
        length_ratio = torch.tensor(item.original_len / self.target_length, dtype=torch.float32)

        sample = {
            "noise": noise,
            "type_id": type_id,
            "length": length,
            "length_ratio": length_ratio,
            "source_id": source_id,
        }
        if self.return_metadata:
            sample["path"] = str(item.path)
            sample["type_name"] = item.type_name
            sample["source_name"] = item.source_name
            sample["offset"] = offset
        return sample

    def get_info(self):
        return {
            "n_samples": len(self.items),
            "target_length": self.target_length,
            "normalize": self.normalize,
            "type_distribution": self.get_type_distribution(),
            "source_distribution": self.get_source_distribution(),
            "length_distribution": self.get_length_distribution(),
        }

    def get_type_distribution(self):
        counts = Counter(item.type_name for item in self.items)
        return {name: counts.get(name, 0) for name in NOISE_TYPE_NAMES}

    def get_source_distribution(self):
        return dict(Counter(item.source_name for item in self.items))

    def get_length_distribution(self):
        return dict(sorted(Counter(item.original_len for item in self.items).items()))


def make_noise_sampler(
    dataset: NoiseNPYDataset,
    balance_type=True,
    balance_source=True,
    real_weight=1.0,
    synthetic_weight=1.0,
    num_samples=None,
):
    """Weighted sampler for class/source-balanced training."""
    type_counts = Counter(item.type_id for item in dataset.items)
    source_counts = Counter(item.source_id for item in dataset.items)
    source_scale = {
        SOURCE_TO_ID["real"]: float(real_weight),
        SOURCE_TO_ID["synthetic"]: float(synthetic_weight),
    }

    weights = []
    for item in dataset.items:
        w = 1.0
        if balance_type:
            w *= 1.0 / max(type_counts[item.type_id], 1)
        if balance_source:
            w *= 1.0 / max(source_counts[item.source_id], 1)
        w *= source_scale.get(item.source_id, 1.0)
        weights.append(w)

    if num_samples is None:
        num_samples = len(dataset)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
    )


if __name__ == "__main__":
    ds = NoiseNPYDataset()
    print(ds.get_info())
    sample = ds[0]
    print(sample["noise"].shape, sample["type_id"], sample["length_ratio"])
