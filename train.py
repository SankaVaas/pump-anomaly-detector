"""
train.py — Train the pump vibration autoencoder.

Local CPU (dev/test):
    python train.py --epochs 5 --batch-size 32 --no-amp

Colab T4 (full training):
    python train.py --epochs 100 --batch-size 128 --drive-dir /content/drive/MyDrive/pump-anomaly

Resume after Colab disconnect:
    python train.py --epochs 100 --resume
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from models.autoencoder import PumpAutoencoder
from utils.dataset import (
    generate_synthetic_vibration,
    generate_anomalous_vibration,
    build_dataloaders,
    load_csv_signal,
    VibrationDataset,
)
from utils.trainer import Trainer
from utils.detector import AnomalyDetector, export_onnx


def parse_args():
    p = argparse.ArgumentParser(description="Train pump vibration autoencoder")
    p.add_argument("--data-csv", type=str, default=None,
                   help="Path to CSV with real vibration data (column: 'vibration'). "
                        "If omitted, synthetic data is used.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--accumulation-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch * steps)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--percentile", type=float, default=99.0,
                   help="Percentile for anomaly threshold calibration")
    p.add_argument("--checkpoint-dir", type=str, default="models/saved")
    p.add_argument("--drive-dir", type=str, default=None,
                   help="Google Drive path for checkpoint mirroring (Colab)")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    p.add_argument("--export-onnx", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mirror_to_drive(src: str, drive_dir: str):
    """Copy checkpoint files to Google Drive for persistence across Colab sessions."""
    import shutil
    drive_path = Path(drive_dir)
    drive_path.mkdir(parents=True, exist_ok=True)
    for f in Path(src).glob("*.pt"):
        shutil.copy2(f, drive_path / f.name)
    for f in Path(src).glob("*.json"):
        shutil.copy2(f, drive_path / f.name)
    print(f"  Mirrored checkpoints → {drive_dir}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if args.data_csv:
        print(f"Loading real data from {args.data_csv}")
        normal_signal = load_csv_signal(args.data_csv)
    else:
        print("No CSV provided — using synthetic vibration data")
        normal_signal = generate_synthetic_vibration(n_samples=300_000, seed=args.seed)

    print(f"Normal signal: {len(normal_signal):,} samples")

    train_loader, val_loader = build_dataloaders(
        normal_signal,
        window_size=args.window_size,
        stride=args.window_size // 4,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = PumpAutoencoder(input_length=args.window_size, latent_dim=args.latent_dim)
    print(f"\nModel: {model.num_parameters:,} parameters")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.amp,
    )

    trainer.train(max_epochs=args.epochs, patience=args.patience, resume=args.resume)

    if args.drive_dir:
        mirror_to_drive(args.checkpoint_dir, args.drive_dir)

    # ------------------------------------------------------------------
    # 4. Calibrate anomaly threshold on validation (normal) data
    # ------------------------------------------------------------------
    print("\nCalibrating anomaly threshold...")
    # Load the best checkpoint for calibration
    trainer.load_checkpoint("best")
    detector = AnomalyDetector(model)
    detector.calibrate(val_loader, percentile=args.percentile)
    threshold_path = Path(args.checkpoint_dir) / "threshold.json"
    detector.save_threshold(str(threshold_path))

    # ------------------------------------------------------------------
    # 5. Quick evaluation on synthetic anomaly data
    # ------------------------------------------------------------------
    print("\nEvaluating on synthetic anomalies...")
    anomaly_signal = np.concatenate([
        normal_signal[-50_000:],
        generate_anomalous_vibration(50_000, anomaly_type="bearing"),
    ])
    labels = np.array(
        [0] * (50_000 // (args.window_size // 4) - 1)
        + [1] * (50_000 // (args.window_size // 4) - 1)
    )
    # Build a combined eval loader
    from torch.utils.data import DataLoader
    eval_ds = VibrationDataset(anomaly_signal, args.window_size, args.window_size // 4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    labels = labels[: len(eval_ds)]  # trim to match dataset length

    try:
        metrics = detector.evaluate(eval_loader, labels)
    except Exception as e:
        print(f"(Evaluation skipped: {e})")

    # ------------------------------------------------------------------
    # 6. Export to ONNX for CPU inference
    # ------------------------------------------------------------------
    if args.export_onnx:
        onnx_path = Path(args.checkpoint_dir) / "pump_autoencoder.onnx"
        export_onnx(model, args.window_size, str(onnx_path))
        if args.drive_dir:
            mirror_to_drive(args.checkpoint_dir, args.drive_dir)

    print("\nDone. Artifacts saved:")
    for f in Path(args.checkpoint_dir).iterdir():
        print(f"  {f}")


if __name__ == "__main__":
    main()
