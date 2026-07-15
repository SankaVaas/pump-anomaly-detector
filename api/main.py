"""
api/main.py — FastAPI backend for pump vibration anomaly detection.

Endpoints:
  GET  /health              — liveness check, returns model status
  GET  /threshold           — returns current threshold value
  POST /score/window        — score a single 512-sample window
  POST /score/batch         — score a list of windows
  POST /score/session       — score one window and append to session history
  GET  /session             — get full session history
  DELETE /session           — clear session history

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from pathlib import Path
import numpy as np
import json
import time
import onnxruntime as ort

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 512
ONNX_PATH     = Path("models/saved/pump_autoencoder.onnx")
THRESHOLD_PATH = Path("models/saved/threshold.json")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pump Anomaly Detector API",
    description=(
        "Convolutional denoising autoencoder for bearing/pump vibration anomaly detection. "
        "Trained on CWRU bearing dataset. Mean recall 0.983 across 11 conditions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model state (loaded once at startup) ──────────────────────────────────────
class ModelState:
    session:   Optional[ort.InferenceSession] = None
    threshold: float = 0.330
    loaded:    bool  = False
    error:     Optional[str] = None

state = ModelState()

# In-memory session history (cleared on restart)
_session_history: List[dict] = []


@app.on_event("startup")
def load_model():
    if not ONNX_PATH.exists():
        state.error = f"ONNX model not found at {ONNX_PATH}. Run training first."
        return
    if not THRESHOLD_PATH.exists():
        state.error = f"Threshold file not found at {THRESHOLD_PATH}. Run training first."
        return
    try:
        state.session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        state.threshold = json.loads(THRESHOLD_PATH.read_text())["threshold"]
        state.loaded    = True
        print(f"Model loaded. Threshold: {state.threshold:.6f}")
    except Exception as e:
        state.error = str(e)
        print(f"Model load failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _score_window(values: np.ndarray) -> float:
    """
    Normalize a single window and return reconstruction MSE.
    Raises RuntimeError if model is not loaded.
    """
    if not state.loaded:
        raise RuntimeError(state.error or "Model not loaded")

    w     = values.astype(np.float32)
    w     = (w - w.mean()) / (w.std() + 1e-8)
    x     = w.reshape(1, 1, -1)
    x_hat = state.session.run(None, {"vibration": x})[0]
    return float(np.mean((x - x_hat) ** 2))


def _make_result(score: float, window_index: Optional[int] = None) -> dict:
    return {
        "score":       round(score, 6),
        "threshold":   round(state.threshold, 6),
        "is_anomaly":  score > state.threshold,
        "margin":      round(score - state.threshold, 6),
        **({"window_index": window_index} if window_index is not None else {}),
    }


# ── Schemas ───────────────────────────────────────────────────────────────────
class WindowRequest(BaseModel):
    """Single 512-sample vibration window."""
    samples: List[float] = Field(
        ...,
        min_items=WINDOW_SIZE,
        max_items=WINDOW_SIZE,
        description=f"Exactly {WINDOW_SIZE} vibration amplitude values (float).",
        example=[0.012, -0.034, 0.056] + [0.0] * 509,
    )
    label: Optional[str] = Field(
        None,
        description="Optional human-readable label for this reading (e.g. 'T+5min').",
    )

    @validator("samples")
    def must_be_finite(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("All samples must be finite numbers (no NaN or Inf).")
        return v


class BatchRequest(BaseModel):
    """Multiple pre-split windows submitted together."""
    windows: List[List[float]] = Field(
        ...,
        description=f"List of windows, each exactly {WINDOW_SIZE} floats.",
    )

    @validator("windows")
    def validate_windows(cls, v):
        for i, w in enumerate(v):
            if len(w) != WINDOW_SIZE:
                raise ValueError(
                    f"Window {i} has {len(w)} samples. Expected {WINDOW_SIZE}."
                )
            if any(not np.isfinite(x) for x in w):
                raise ValueError(f"Window {i} contains NaN or Inf values.")
        return v


class RawSamplesRequest(BaseModel):
    """
    Raw contiguous samples — the API splits them into windows automatically.
    Any trailing samples that don't fill a complete window are discarded.
    """
    samples: List[float] = Field(
        ...,
        min_items=WINDOW_SIZE,
        description=f"Contiguous vibration samples. Min {WINDOW_SIZE}. "
                    "Will be split into non-overlapping windows of 512.",
    )

    @validator("samples")
    def must_be_finite(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("All samples must be finite numbers.")
        return v


class SessionWindow(BaseModel):
    """Single window added to the live session timeline."""
    samples: List[float] = Field(..., min_items=WINDOW_SIZE, max_items=WINDOW_SIZE)
    label: Optional[str] = None

    @validator("samples")
    def must_be_finite(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("All samples must be finite numbers.")
        return v


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health():
    """Liveness check. Returns model load status and threshold."""
    return {
        "status":     "ok" if state.loaded else "degraded",
        "model":      "loaded" if state.loaded else "not loaded",
        "threshold":  state.threshold,
        "error":      state.error,
        "window_size": WINDOW_SIZE,
    }


@app.get("/threshold", tags=["Status"])
def get_threshold():
    """Returns the current anomaly threshold."""
    return {"threshold": state.threshold}


@app.post("/threshold", tags=["Status"])
def set_threshold(value: float):
    """
    Override the anomaly threshold at runtime.
    Useful for tuning precision/recall tradeoff without restarting.
    Min: 0.001. Max: 10.0.
    """
    if not (0.001 <= value <= 10.0):
        raise HTTPException(status_code=422, detail="Threshold must be between 0.001 and 10.0")
    state.threshold = value
    return {"threshold": state.threshold, "message": "Threshold updated"}


@app.post("/score/window", tags=["Scoring"])
def score_single_window(req: WindowRequest):
    """
    Score a single 512-sample vibration window.

    Returns the reconstruction MSE, whether it exceeds the threshold,
    and the margin (positive = anomaly, negative = normal).
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "Model not loaded")
    try:
        t0    = time.perf_counter()
        score = _score_window(np.array(req.samples))
        ms    = (time.perf_counter() - t0) * 1000
        return {**_make_result(score), "inference_ms": round(ms, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", tags=["Scoring"])
def score_batch(req: BatchRequest):
    """
    Score multiple pre-split windows in one call.

    Returns a result for each window plus aggregate stats.
    Use this for scoring a full recording at once.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "Model not loaded")
    try:
        t0      = time.perf_counter()
        results = []
        for i, w in enumerate(req.windows):
            score = _score_window(np.array(w))
            results.append(_make_result(score, window_index=i))
        ms        = (time.perf_counter() - t0) * 1000
        scores    = [r["score"] for r in results]
        n_anomaly = sum(1 for r in results if r["is_anomaly"])
        return {
            "windows":      results,
            "summary": {
                "total_windows":   len(results),
                "anomaly_windows": n_anomaly,
                "normal_windows":  len(results) - n_anomaly,
                "anomaly_rate":    round(n_anomaly / len(results), 4),
                "mean_score":      round(float(np.mean(scores)), 6),
                "max_score":       round(float(np.max(scores)),  6),
                "min_score":       round(float(np.min(scores)),  6),
            },
            "inference_ms": round(ms, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/raw", tags=["Scoring"])
def score_raw_samples(req: RawSamplesRequest):
    """
    Score contiguous raw samples — the API splits them into windows automatically.

    Provide any number of samples (min 512). Trailing samples that don't
    fill a complete 512-sample window are discarded and reported.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "Model not loaded")
    try:
        arr      = np.array(req.samples, dtype=np.float32)
        n_win    = len(arr) // WINDOW_SIZE
        leftover = len(arr)  % WINDOW_SIZE

        t0      = time.perf_counter()
        results = []
        for i in range(n_win):
            window = arr[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
            score  = _score_window(window)
            results.append(_make_result(score, window_index=i))
        ms = (time.perf_counter() - t0) * 1000

        scores    = [r["score"] for r in results]
        n_anomaly = sum(1 for r in results if r["is_anomaly"])
        return {
            "windows":          results,
            "samples_received": len(arr),
            "samples_used":     n_win * WINDOW_SIZE,
            "samples_discarded":leftover,
            "summary": {
                "total_windows":   n_win,
                "anomaly_windows": n_anomaly,
                "normal_windows":  n_win - n_anomaly,
                "anomaly_rate":    round(n_anomaly / n_win, 4) if n_win else 0,
                "mean_score":      round(float(np.mean(scores)), 6) if scores else 0,
                "max_score":       round(float(np.max(scores)),  6) if scores else 0,
            },
            "inference_ms": round(ms, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/add", tags=["Session"])
def session_add(req: SessionWindow):
    """
    Add one window to the live session timeline.

    Each call appends a scored entry. The session persists until
    DELETE /session is called or the server restarts.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "Model not loaded")
    try:
        score = _score_window(np.array(req.samples))
        entry = {
            "index":      len(_session_history),
            "label":      req.label or f"Reading {len(_session_history) + 1}",
            "timestamp":  time.time(),
            **_make_result(score),
        }
        _session_history.append(entry)
        return entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session", tags=["Session"])
def session_get():
    """
    Return the full live session timeline with aggregate stats.
    """
    if not _session_history:
        return {"readings": [], "summary": None}

    scores    = [e["score"] for e in _session_history]
    n_anomaly = sum(1 for e in _session_history if e["is_anomaly"])
    return {
        "readings": _session_history,
        "summary": {
            "total":       len(_session_history),
            "anomalies":   n_anomaly,
            "normal":      len(_session_history) - n_anomaly,
            "mean_score":  round(float(np.mean(scores)), 6),
            "max_score":   round(float(np.max(scores)),  6),
            "threshold":   state.threshold,
        },
    }


@app.delete("/session", tags=["Session"])
def session_clear():
    """Clear all session history."""
    _session_history.clear()
    return {"message": "Session cleared", "readings": 0}