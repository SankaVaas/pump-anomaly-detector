import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path
from typing import Tuple, Optional
import json


class AnomalyDetector:
    """
    Wraps a trained autoencoder and a calibrated reconstruction-error threshold.

    Threshold is computed from normal validation data at a chosen percentile
    (e.g. 99th) so that ~1% of normal windows trigger a false alarm.

    Usage:
        detector = AnomalyDetector(model, device)
        detector.calibrate(normal_loader, percentile=99)
        detector.save_threshold("models/saved/threshold.json")

        # Later, for inference:
        detector.load_threshold("models/saved/threshold.json")
        scores, flags = detector.predict(new_loader)
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_scores(self, loader: DataLoader, use_amp: bool = True) -> np.ndarray:
        """Return per-window reconstruction errors (MSE) for an entire loader."""
        errors = []
        amp_enabled = use_amp and self.device.type == "cuda"
        for batch in loader:
            x = batch.to(self.device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                err = self.model.reconstruction_error(x)
            errors.append(err.cpu().numpy())
        return np.concatenate(errors)

    def calibrate(
        self,
        normal_loader: DataLoader,
        percentile: float = 99.0,
        use_amp: bool = True,
    ) -> float:
        """
        Set threshold = `percentile`-th percentile of normal reconstruction errors.
        A window is flagged as anomalous if its error exceeds this threshold.
        """
        print(f"Calibrating threshold on {len(normal_loader.dataset)} normal windows...")
        scores = self.compute_scores(normal_loader, use_amp)
        self.threshold = float(np.percentile(scores, percentile))
        print(f"Threshold set at {percentile}th percentile: {self.threshold:.6f}")
        print(f"  mean={scores.mean():.6f}  std={scores.std():.6f}  "
              f"max={scores.max():.6f}")
        return self.threshold

    def save_threshold(self, path: str):
        assert self.threshold is not None, "Call calibrate() first"
        with open(path, "w") as f:
            json.dump({"threshold": self.threshold}, f, indent=2)
        print(f"Threshold saved → {path}")

    def load_threshold(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.threshold = data["threshold"]
        print(f"Threshold loaded: {self.threshold:.6f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        use_amp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            scores: reconstruction error per window  (shape: N,)
            flags:  boolean anomaly flag per window  (shape: N,)
        """
        assert self.threshold is not None, "Set threshold via calibrate() or load_threshold()"
        scores = self.compute_scores(loader, use_amp)
        flags = scores > self.threshold
        return scores, flags

    def predict_single(self, window: np.ndarray) -> Tuple[float, bool]:
        """
        Predict on a single raw numpy window (shape: window_size,).
        Suitable for real-time inference on CPU after ONNX or PyTorch export.
        """
        assert self.threshold is not None
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            err = self.model.reconstruction_error(x).item()
        return err, err > self.threshold

    # ------------------------------------------------------------------
    # Evaluation metrics (requires ground-truth labels)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        labels: np.ndarray,
        use_amp: bool = True,
    ) -> dict:
        """
        Compute precision, recall, F1, and AUC-ROC.
        `labels`: 0 = normal, 1 = anomaly, length must match number of windows.
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            confusion_matrix,
        )

        scores, flags = self.predict(loader, use_amp)
        preds = flags.astype(int)

        metrics = {
            "threshold": self.threshold,
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(labels, scores),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }
        print("\n--- Evaluation Results ---")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")
        tn, fp, fn, tp = np.array(metrics["confusion_matrix"]).ravel()
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        return metrics


def export_onnx(model: nn.Module, input_length: int, output_path: str):
    """
    Export trained model to ONNX for fast CPU inference (no PyTorch needed).
    ONNX Runtime inference is 2–5x faster than native PyTorch CPU.
    """
    model.eval().cpu()
    dummy = torch.randn(1, 1, input_length)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["vibration"],
        output_names=["reconstruction"],
        dynamic_axes={"vibration": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported → {output_path}")


class ONNXInferenceEngine:
    """Lightweight CPU inference using ONNX Runtime. No GPU or PyTorch needed."""

    def __init__(self, onnx_path: str, threshold_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        with open(threshold_path) as f:
            self.threshold = json.load(f)["threshold"]
        print(f"ONNX engine ready. Threshold: {self.threshold:.6f}")

    def predict(self, window: np.ndarray) -> Tuple[float, bool]:
        x = window.astype(np.float32).reshape(1, 1, -1)
        x_hat = self.session.run(None, {"vibration": x})[0]
        err = float(np.mean((x - x_hat) ** 2))
        return err, err > self.threshold
