"""
Smoke tests for the API and preprocessing.

Run:  pytest -q
(These do NOT require a trained model or the full dataset — they check that the
app boots, routes respond, and single-image preprocessing produces the right shape.)
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app
from src.preprocessing import preprocess_image
from src.config import IMG_HEIGHT, IMG_WIDTH, CHANNELS

client = TestClient(app)


def _fake_image_bytes():
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status_shape():
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "model_available", "uptime_seconds", "prediction_count"):
        assert key in body


def test_preprocess_shape():
    arr = preprocess_image(_fake_image_bytes())
    assert arr.shape == (1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_predict_without_model_returns_503():
    # With no trained model present, /predict should fail gracefully (not crash).
    files = {"file": ("x.jpg", _fake_image_bytes(), "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code in (200, 503)  # 503 if untrained, 200 if a model exists
