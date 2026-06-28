import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal
from pathlib import Path
from typing import Tuple, Optional
import json


class VibrationDataset(Dataset):
    """
    Sliding-window dataset over a 1-D vibration signal array.

    Each sample is a window of `window_size` samples.
    Stride controls overlap (stride < window_size = overlapping windows).
    """

    def __init__(
        self,
        signal: np.ndarray,
        window_size: int = 512,
        stride: int = 256,
        normalize: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride

        if normalize:
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        # Build index of (start,) positions
        starts = np.arange(0, len(signal) - window_size + 1, stride)
        self.windows = np.stack(
            [signal[s : s + window_size] for s in starts], axis=0
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Shape: (1, window_size) — channel-first for Conv1d
        return torch.from_numpy(self.windows[idx]).unsqueeze(0)


def generate_synthetic_vibration(
    n_samples: int = 200_000,
    sample_rate: int = 10_000,
    normal_freqs: Tuple[float, ...] = (50.0, 150.0, 300.0),
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic 'normal' pump vibration — sum of harmonics + noise.
    Useful for development when real sensor data is unavailable.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)
    for freq in normal_freqs:
        amplitude = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    signal += rng.normal(0, noise_std, n_samples)
    return signal.astype(np.float32)


def generate_anomalous_vibration(
    n_samples: int = 50_000,
    sample_rate: int = 10_000,
    anomaly_type: str = "bearing",
    seed: int = 99,
) -> np.ndarray:
    """
    Generate synthetic anomalous vibration for threshold evaluation only —
    never used during training.

    anomaly_type options:
      'bearing' — impulse-like bearing defect signature
      'imbalance' — strong single frequency imbalance
      'cavitation' — broadband noise burst
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate
    base = generate_synthetic_vibration(n_samples, sample_rate, seed=seed + 1)

    if anomaly_type == "bearing":
        defect_freq = 105.0
        impulse = 3.0 * np.sin(2 * np.pi * defect_freq * t)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))
        return (base + impulse * envelope).astype(np.float32)

    elif anomaly_type == "imbalance":
        return (base + 4.0 * np.sin(2 * np.pi * 25.0 * t)).astype(np.float32)

    elif anomaly_type == "cavitation":
        burst = rng.normal(0, 1.5, n_samples)
        burst_mask = np.zeros(n_samples)
        for _ in range(20):
            start = rng.integers(0, n_samples - 2000)
            burst_mask[start : start + 2000] = 1.0
        return (base + burst * burst_mask).astype(np.float32)

    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")


def build_dataloaders(
    normal_signal: np.ndarray,
    window_size: int = 512,
    stride: int = 128,
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split normal signal into train/val dataloaders."""
    split = int(len(normal_signal) * (1 - val_split))
    train_sig = normal_signal[:split]
    val_sig = normal_signal[split:]

    train_ds = VibrationDataset(train_sig, window_size, stride)
    val_ds = VibrationDataset(val_sig, window_size, stride)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def load_csv_signal(path: str, column: str = "vibration") -> np.ndarray:
    """
    Load a single-column vibration CSV.
    CSV must have a header row; use `column` to pick the signal column.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    return df[column].values.astype(np.float32)
