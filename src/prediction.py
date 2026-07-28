"""
prediction.py
=============
Thin, dependency-light wrapper around the trained model used by the API.

The model is loaded once and cached (lazy singleton) so that repeated
prediction requests — including a Locust flood — do not reload it from disk.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.config import CLASS_NAMES, MODEL_PATH, MODEL_PATH_H5
from src.preprocessing import preprocess_image

_MODEL = None  # module-level cache


def model_is_available() -> bool:
    return MODEL_PATH.exists() or MODEL_PATH_H5.exists()


def get_model():
    """Lazy-load and cache the trained Keras model."""
    global _MODEL
    if _MODEL is None:
        from src.model import load_model
        _MODEL = load_model()
    return _MODEL


def reset_model_cache() -> None:
    """Force the next prediction to reload the model (used after retraining)."""
    global _MODEL
    _MODEL = None


def predict(source) -> Dict:
    """
    Predict pneumonia vs normal for a single image.

    ``source`` can be a file path or raw image bytes.
    Returns a dict with the predicted label, confidence and per-class probabilities.
    """
    model = get_model()
    batch = preprocess_image(source)
    prob_pneumonia = float(model.predict(batch, verbose=0).ravel()[0])
    prob_normal = 1.0 - prob_pneumonia

    label_idx = int(prob_pneumonia >= 0.5)
    label = CLASS_NAMES[label_idx]
    confidence = prob_pneumonia if label_idx == 1 else prob_normal

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "NORMAL": round(prob_normal, 4),
            "PNEUMONIA": round(prob_pneumonia, 4),
        },
    }
