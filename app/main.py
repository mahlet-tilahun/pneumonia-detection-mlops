"""
main.py — FastAPI application
=============================
Serves BOTH the JSON API and the single-page web UI, so the whole solution runs
as one container behind one public URL (simple to deploy and to flood with Locust).

Endpoints
---------
GET  /                 -> the web UI (prediction, visualisations, upload, retrain)
GET  /health           -> liveness probe
GET  /status           -> uptime, model status, prediction & upload counters
POST /predict          -> predict one image  (used by the UI and by Locust)
GET  /visualizations   -> dataset summary powering the 3 feature charts
GET  /metrics          -> latest evaluation metrics of the deployed model
POST /upload           -> bulk-upload labelled images for retraining
POST /retrain          -> trigger a retraining run (background task)
GET  /retrain/status   -> progress of the current/last retraining run
"""
from __future__ import annotations

import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import (
    CLASS_NAMES, UPLOAD_DIR, TRAIN_DIR, TEST_DIR, METRICS_PATH,
    RETRAIN_TRIGGER_THRESHOLD, ensure_dirs,
)
from src.preprocessing import preprocess_upload_dir, dataset_summary
from src import prediction as pred

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = APP_DIR / "templates"

app = FastAPI(title="Pneumonia Detection MLOps", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory runtime state (uptime + monitoring counters for the UI)
# ---------------------------------------------------------------------------
STATE = {
    "start_time": time.time(),
    "prediction_count": 0,
    "upload_count": 0,
    "last_prediction_at": None,
    "retrain": {"running": False, "message": "idle", "last_result": None,
                "started_at": None, "finished_at": None},
}
_retrain_lock = threading.Lock()


@app.on_event("startup")
def _startup():
    ensure_dirs()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    uptime = time.time() - STATE["start_time"]
    return {
        "status": "online",
        "model_available": pred.model_is_available(),
        "uptime_seconds": round(uptime, 1),
        "uptime_human": _human_uptime(uptime),
        "prediction_count": STATE["prediction_count"],
        "upload_count": STATE["upload_count"],
        "last_prediction_at": STATE["last_prediction_at"],
        "pending_uploads": preprocess_upload_dir(UPLOAD_DIR),
        "retrain": STATE["retrain"],
        "server_time": datetime.utcnow().isoformat() + "Z",
    }


def _human_uptime(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Prediction  (single datapoint from an image)
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not pred.model_is_available():
        raise HTTPException(status_code=503,
                            detail="Model not trained yet. Run training or retraining first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    try:
        result = pred.predict(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    STATE["prediction_count"] += 1
    STATE["last_prediction_at"] = datetime.utcnow().isoformat() + "Z"
    result["filename"] = file.filename
    return result


# ---------------------------------------------------------------------------
# Visualisations / metrics
# ---------------------------------------------------------------------------
@app.get("/visualizations")
def visualizations():
    """Dataset summary powering the 3 interpreted feature charts in the UI."""
    src_dir = TRAIN_DIR if any(TRAIN_DIR.glob("*/*")) else TEST_DIR
    summary = dataset_summary(src_dir)
    return JSONResponse(summary)


@app.get("/metrics")
def metrics():
    if METRICS_PATH.exists():
        return JSONResponse(json.loads(METRICS_PATH.read_text()))
    return JSONResponse({"detail": "No evaluation metrics yet. Train the model first."},
                        status_code=404)


# ---------------------------------------------------------------------------
# Bulk upload of labelled images for retraining
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), label: str = Form(...)):
    label = label.upper().strip()
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400,
                            detail=f"label must be one of {CLASS_NAMES}")
    ensure_dirs()
    dest = UPLOAD_DIR / label
    dest.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            continue
        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        (dest / f"{stamp}_{f.filename}").write_bytes(await f.read())
        saved += 1

    STATE["upload_count"] += saved
    counts = preprocess_upload_dir(UPLOAD_DIR)

    # Automatic retraining trigger: fire once enough new data has accumulated.
    auto_triggered = counts["total"] >= RETRAIN_TRIGGER_THRESHOLD
    return {
        "status": "success",
        "saved": saved,
        "label": label,
        "pending_uploads": counts,
        "retrain_threshold": RETRAIN_TRIGGER_THRESHOLD,
        "auto_retrain_recommended": auto_triggered,
    }


# ---------------------------------------------------------------------------
# Retraining trigger
# ---------------------------------------------------------------------------
def _run_retrain(model_type: str, epochs: int):
    from src.model import retrain
    try:
        STATE["retrain"].update({"running": True, "message": "retraining in progress...",
                                 "started_at": datetime.utcnow().isoformat() + "Z",
                                 "finished_at": None})
        result = retrain(model_type=model_type, epochs=epochs)
        pred.reset_model_cache()  # serve the freshly trained model
        STATE["retrain"].update({"running": False, "message": "completed",
                                 "last_result": result,
                                 "finished_at": datetime.utcnow().isoformat() + "Z"})
    except Exception as e:
        STATE["retrain"].update({"running": False, "message": f"failed: {e}",
                                 "finished_at": datetime.utcnow().isoformat() + "Z"})
    finally:
        if _retrain_lock.locked():
            _retrain_lock.release()


@app.post("/retrain")
def retrain_endpoint(background_tasks: BackgroundTasks,
                     model_type: str = Form("mobilenet"),
                     epochs: int = Form(8)):
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining run is already in progress.")
    background_tasks.add_task(_run_retrain, model_type, int(epochs))
    return {"status": "started",
            "message": "Retraining triggered. Poll /retrain/status for progress.",
            "model_type": model_type, "epochs": int(epochs)}


@app.get("/retrain/status")
def retrain_status():
    return STATE["retrain"]
