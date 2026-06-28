"""
infer.py — Real-time pump anomaly inference on local CPU.

Uses ONNX Runtime (no GPU, no PyTorch needed at deployment).

Stream from CSV file:
    python infer.py --source data/raw/live_sensor.csv

Stream from stdin (pipe from sensor reader):
    python infer.py --source stdin

Demo mode (synthetic signal):
    python infer.py --demo
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Real-time pump anomaly inference")
    p.add_argument("--model", type=str, default="models/saved/pump_autoencoder.onnx")
    p.add_argument("--threshold", type=str, default="models/saved/threshold.json")
    p.add_argument("--source", type=str, default=None,
                   help="CSV path or 'stdin'")
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--column", type=str, default="vibration")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--demo-anomaly", type=str, default="bearing",
                   choices=["bearing", "imbalance", "cavitation"])
    p.add_argument("--alert-threshold", type=int, default=3,
                   help="Consecutive anomalous windows before alert")
    return p.parse_args()


def run_demo(engine, window_size: int, anomaly_type: str, alert_threshold: int):
    from utils.dataset import generate_synthetic_vibration, generate_anomalous_vibration

    print(f"\nDemo mode | anomaly type: {anomaly_type}")
    print(f"{'Window':>8}  {'Score':>12}  {'Status':>10}  {'Note'}")
    print("-" * 55)

    # Simulate: 20 normal windows, then 20 anomalous
    normal_sig = generate_synthetic_vibration(window_size * 20 + 512)
    anomaly_sig = generate_anomalous_vibration(window_size * 20 + 512, anomaly_type=anomaly_type)
    full_sig = np.concatenate([normal_sig, anomaly_sig])

    consecutive = 0
    for i in range(0, len(full_sig) - window_size, window_size):
        window = full_sig[i : i + window_size]
        # Normalize window
        window = (window - window.mean()) / (window.std() + 1e-8)
        score, is_anomaly = engine.predict(window)

        phase = "NORMAL region" if i < window_size * 20 else "ANOMALY region"
        status = "ANOMALY ⚠" if is_anomaly else "normal"
        consecutive = (consecutive + 1) if is_anomaly else 0

        print(f"{i//window_size:>8}  {score:>12.6f}  {status:>10}  {phase}")

        if consecutive >= alert_threshold:
            print(f"\n  *** ALERT: {consecutive} consecutive anomalous windows detected ***\n")
            consecutive = 0

        time.sleep(0.05)  # simulate real-time rate


def run_csv(engine, csv_path: str, column: str, window_size: int, alert_threshold: int):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        print(f"Column '{column}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    signal = df[column].values.astype(np.float32)

    print(f"\nProcessing {csv_path} | {len(signal)} samples | "
          f"{len(signal)//window_size} windows")
    print(f"{'Window':>8}  {'Score':>12}  {'Anomaly?':>10}")
    print("-" * 40)

    consecutive = 0
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i : i + window_size]
        window = (window - window.mean()) / (window.std() + 1e-8)
        score, is_anomaly = engine.predict(window)
        consecutive = (consecutive + 1) if is_anomaly else 0
        flag = "ANOMALY ⚠" if is_anomaly else "ok"
        print(f"{i//window_size:>8}  {score:>12.6f}  {flag:>10}")
        if consecutive >= alert_threshold:
            print(f"\n  *** ALERT: {consecutive} consecutive anomalous windows ***\n")
            consecutive = 0


def run_stdin(engine, window_size: int, alert_threshold: int):
    """
    Read comma-separated float values from stdin, one sample per line.
    Accumulate into windows and predict.
    """
    print(f"Reading from stdin. Window size: {window_size}. Ctrl+C to stop.")
    buf = []
    consecutive = 0
    for line in sys.stdin:
        try:
            vals = [float(v) for v in line.strip().split(",")]
            buf.extend(vals)
        except ValueError:
            continue

        while len(buf) >= window_size:
            window = np.array(buf[:window_size], dtype=np.float32)
            buf = buf[window_size:]
            window = (window - window.mean()) / (window.std() + 1e-8)
            score, is_anomaly = engine.predict(window)
            consecutive = (consecutive + 1) if is_anomaly else 0
            ts = time.strftime("%H:%M:%S")
            flag = "ANOMALY ⚠" if is_anomaly else "normal"
            print(f"[{ts}]  score={score:.6f}  status={flag}")
            if consecutive >= alert_threshold:
                print(f"*** ALERT: {consecutive} consecutive anomalous windows ***")
                consecutive = 0


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"ONNX model not found: {args.model}")
        print("Train first: python train.py")
        sys.exit(1)

    from utils.detector import ONNXInferenceEngine
    engine = ONNXInferenceEngine(args.model, args.threshold)

    if args.demo:
        run_demo(engine, args.window_size, args.demo_anomaly, args.alert_threshold)
    elif args.source == "stdin":
        run_stdin(engine, args.window_size, args.alert_threshold)
    elif args.source:
        run_csv(engine, args.source, args.column, args.window_size, args.alert_threshold)
    else:
        print("Specify --demo, --source <file.csv>, or --source stdin")
        sys.exit(1)


if __name__ == "__main__":
    main()
