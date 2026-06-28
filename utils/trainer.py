import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional, Dict, List
import json
import time


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Training engine for the pump autoencoder.

    Features:
      - Mixed precision (fp16) via torch.cuda.amp — halves VRAM usage
      - Gradient accumulation — simulates large batches on 12 GB
      - Google Drive checkpointing — survive Colab disconnects
      - Early stopping
      - Session-resumable training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "models/saved",
        accumulation_steps: int = 4,
        use_amp: bool = True,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and self.device.type == "cuda"

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Grad accumulation: {accumulation_steps} steps")
        print(f"Effective batch size: {train_loader.batch_size * accumulation_steps}")
        print(f"Model parameters: {model.num_parameters:,}")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "last"):
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            path,
        )
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, tag: str = "last") -> bool:
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        if not path.exists():
            print(f"No checkpoint found at {path}. Starting from scratch.")
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.history = ckpt["history"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best val loss: {self.best_val_loss:.6f})")
        return True

    # ------------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            x = batch.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                x_hat = self.model(x)
                loss = self.criterion(x_hat, x) / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(
                self.train_loader
            ):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            # In Trainer._train_epoch(), replace:
            #   x = batch.to(self.device)
            #   x_hat = self.model(x)
            #   loss = self.criterion(x_hat, x)
            #
            # With:

            x_noisy, x_clean = batch[0].to(self.device), batch[1].to(self.device)
            x_hat = self.model(x_noisy)
            loss = self.criterion(x_hat, x_clean)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        max_epochs: int = 100,
        patience: int = 15,
        resume: bool = True,
    ):
        if resume:
            self.load_checkpoint("last")

        early_stop = EarlyStopping(patience=patience)

        for epoch in range(self.start_epoch, max_epochs):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")

            self.save_checkpoint(epoch, val_loss, tag="last")
            self._save_history()  # save every epoch — survive Colab cutoff

            marker = " *" if improved else ""
            print(
                f"Epoch {epoch+1:03d}/{max_epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | "
                f"{elapsed:.1f}s{marker}"
            )

            if early_stop(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")

    def _save_history(self):
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved → {path}")