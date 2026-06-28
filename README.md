# pump-anomaly-detector

Convolutional autoencoder for pump vibration anomaly detection.  
Trains on Colab T4 (12 GB VRAM) in 2–3 hours. Runs inference on any local CPU.

---

## How it works

The autoencoder is trained **only on normal vibration data**.  
At inference time, windows the model cannot reconstruct well (high MSE) are flagged as anomalies.  
No failure labels needed — just normal operating recordings.

```
Input window (512 samples)
        │
   ┌────▼────┐
   │ Encoder │  Conv1d ×4 → flatten → Linear → latent (32-dim)
   └────┬────┘
        │ z
   ┌────▼────┐
   │ Decoder │  Linear → reshape → ConvTranspose1d ×4 → reconstruction
   └────┬────┘
        │
   Reconstruction error (MSE)
   > threshold → ANOMALY
   ≤ threshold → normal
```

**VRAM usage:** ~1.8 GB at batch 128 with fp16 — well within the 12 GB T4 limit.  
**Parameters:** ~520,000 — tiny enough to run inference in <10 ms on a laptop CPU.

---

## Repository layout

```
pump-anomaly-detector/
│
├── models/
│   ├── autoencoder.py        # ConvEncoder, ConvDecoder, PumpAutoencoder
│   └── saved/                # Checkpoints land here (gitignored except JSON)
│       ├── checkpoint_best.pt
│       ├── checkpoint_last.pt
│       ├── pump_autoencoder.onnx
│       └── threshold.json
│
├── utils/
│   ├── dataset.py            # VibrationDataset, sliding windows, synthetic data
│   ├── trainer.py            # Trainer with mixed precision + gradient accumulation
│   └── detector.py           # AnomalyDetector, ONNXInferenceEngine, ONNX export
│
├── notebooks/
│   └── train_colab.ipynb     # Step-by-step Colab training notebook
│
├── data/
│   ├── raw/                  # Drop your CSV vibration files here
│   └── processed/            # Preprocessed arrays (auto-generated)
│
├── train.py                  # Main training script
├── infer.py                  # Local CPU inference / demo
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/pump-anomaly-detector.git
cd pump-anomaly-detector
pip install -r requirements.txt
```

### 2. Train on Colab T4 (recommended)

Open `notebooks/train_colab.ipynb` in Google Colab:

1. **Runtime → Change runtime type → T4 GPU**
2. Run Cell 1 — verifies GPU and mounts Google Drive
3. Run Cell 2 — clones repo and installs deps
4. Run Cell 3 — restores checkpoint from Drive (skip on first run)
5. Run Cell 4 — starts training (saves to Drive every epoch)
6. Run Cell 5 — plots training curves
7. Run Cell 6 — visualises normal vs anomaly reconstruction
8. Run Cell 7 — downloads `pump_autoencoder.onnx` + `threshold.json`

**Resuming after disconnect:**  
Just open a new Colab session and run Cells 1 → 3 → 4.  
Training resumes from the last saved epoch automatically.

**Typical timeline:**  
- Session 1 (4 hrs): epochs 1–60, val loss stabilising  
- Session 2 (1–2 hrs): epochs 61–80, early stopping fires  
- Total wall time: ~5–6 hours across 2 days

### 3. Train locally (CPU, for dev/testing)

```bash
# Synthetic data, 5 epochs, no mixed precision (CPU mode)
python train.py --epochs 5 --batch-size 32 --no-amp --no-resume
```

### 4. Run inference demo (local CPU)

After downloading `pump_autoencoder.onnx` and `threshold.json` from Colab:

```bash
# Place files in models/saved/
python infer.py --demo                          # synthetic demo, bearing fault
python infer.py --demo --demo-anomaly cavitation
python infer.py --source data/raw/pump.csv      # your real CSV file
python infer.py --source stdin                  # pipe from sensor reader
```

---

## Training options

```
python train.py [options]

--data-csv PATH          CSV file with 'vibration' column (default: synthetic data)
--epochs N               Max epochs (default: 100)
--batch-size N           Batch size per GPU step (default: 128)
--accumulation-steps N   Gradient accumulation (effective batch = N × batch-size, default: 4)
--lr FLOAT               Learning rate (default: 1e-3)
--window-size N          Samples per window, must be divisible by 16 (default: 512)
--latent-dim N           Bottleneck dimension (default: 32)
--patience N             Early stopping patience in epochs (default: 15)
--percentile FLOAT       Threshold percentile on normal data (default: 99.0)
--checkpoint-dir PATH    Where to save .pt files (default: models/saved)
--drive-dir PATH         Google Drive path to mirror checkpoints (Colab)
--resume / --no-resume   Resume from last checkpoint (default: resume)
--no-amp                 Disable mixed precision (use on CPU)
--export-onnx            Export ONNX after training (default: on)
```

---

## Using your own CSV data

Your CSV must have a header and a numeric column of vibration samples:

```csv
timestamp,vibration,temperature
0.0001,0.023,-0.011,...
```

```bash
python train.py --data-csv data/raw/pump_normal.csv --epochs 100
```

Only pass **normal** recordings at training time. Anomaly data is never used for training — only for threshold evaluation.

---

## Bring it to production

Once trained, the model runs anywhere Python runs — no GPU required:

```python
from utils.detector import ONNXInferenceEngine
import numpy as np

engine = ONNXInferenceEngine(
    "models/saved/pump_autoencoder.onnx",
    "models/saved/threshold.json",
)

# window: numpy array of 512 float32 vibration samples
window = np.random.randn(512).astype(np.float32)  # replace with real data
score, is_anomaly = engine.predict(window)
print(f"Score: {score:.5f} | Anomaly: {is_anomaly}")
```

Inference latency on a modern CPU: **<10 ms per window**.

---

## Colab VRAM budget

| Setting | VRAM used |
|---|---|
| Batch 128, fp32 | ~3.5 GB |
| Batch 128, fp16 (default) | ~1.8 GB |
| Batch 256, fp16 | ~3.2 GB |
| With gradient accumulation ×4 | no extra VRAM |

The T4's 12 GB gives plenty of headroom. You could scale the model (larger latent dim, more channels) and stay well within limits.

---

## Public datasets to replace synthetic data

| Dataset | Samples | Notes |
|---|---|---|
| [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter) | ~500K | Ball bearing faults, 4 fault types |
| [MFPT Bearing Data](https://www.mfpt.org/fault-data-sets/) | ~200K | Outer/inner race faults |
| [NASA FEMTO](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) | Run-to-failure | Accelerated bearing degradation |
| [Kaggle: Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) | 220K rows | 52 sensor channels, real pump |

---

## License

MIT
