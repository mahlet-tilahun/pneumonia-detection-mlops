"""
preprocessing.py
================
All data-loading and image-preprocessing logic lives here.

Functions
---------
- build_generators()      -> train/val/test Keras data generators (with augmentation)
- preprocess_image()      -> turn a single raw image (path or bytes) into a model-ready batch
- preprocess_upload_dir() -> validate & count user-uploaded images before retraining
- dataset_summary()       -> class counts + image statistics used by the visualisations
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image

from src.config import (
    IMG_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE, SEED,
    CLASS_NAMES, TRAIN_DIR, TEST_DIR, VAL_DIR,
)


# ---------------------------------------------------------------------------
# 1. Data generators for training / evaluation
# ---------------------------------------------------------------------------
def build_generators(train_dir: Path = TRAIN_DIR,
                     test_dir: Path = TEST_DIR,
                     val_dir: Path = VAL_DIR,
                     batch_size: int = BATCH_SIZE):
    """
    Build augmented training generator and rescaled val/test generators.

    Data augmentation (rotation, zoom, shift, flip) is a form of regularisation
    that reduces over-fitting on the relatively small chest X-ray dataset.
    If a dedicated ``val`` folder is not present we carve a 15 % validation
    split out of the training data instead.
    """
    # Imported here so that importing this module never requires TensorFlow
    # (the FastAPI app can preprocess a single image without training deps).
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    use_val_folder = val_dir.exists() and any(val_dir.iterdir())

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.0 if use_val_folder else 0.15,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    color_mode = "rgb" if CHANNELS == 3 else "grayscale"

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="binary", color_mode=color_mode, classes=CLASS_NAMES,
        shuffle=True, seed=SEED,
        subset=None if use_val_folder else "training",
    )

    if use_val_folder:
        val_gen = eval_datagen.flow_from_directory(
            val_dir, target_size=IMG_SIZE, batch_size=batch_size,
            class_mode="binary", color_mode=color_mode, classes=CLASS_NAMES,
            shuffle=False,
        )
    else:
        val_gen = train_datagen.flow_from_directory(
            train_dir, target_size=IMG_SIZE, batch_size=batch_size,
            class_mode="binary", color_mode=color_mode, classes=CLASS_NAMES,
            shuffle=False, seed=SEED, subset="validation",
        )

    test_gen = eval_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="binary", color_mode=color_mode, classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def compute_class_weights(train_gen) -> Dict[int, float]:
    """
    The dataset is imbalanced (~3x more PNEUMONIA than NORMAL). Class weights
    tell the loss function to pay more attention to the minority class.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(set(train_gen.classes)))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_gen.classes
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# 2. Single-image preprocessing (used by the API for prediction)
# ---------------------------------------------------------------------------
def preprocess_image(source) -> np.ndarray:
    """
    Convert a single image into a normalised (1, H, W, C) float32 batch.

    ``source`` may be a file path (str / Path) or raw bytes (from an upload).
    Returns an array ready to feed straight into ``model.predict``.
    """
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    img = img.convert("RGB" if CHANNELS == 3 else "L")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if CHANNELS == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)   # add batch dimension
    return arr


# ---------------------------------------------------------------------------
# 3. Validate a folder of uploaded images before retraining
# ---------------------------------------------------------------------------
VALID_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}


def preprocess_upload_dir(upload_dir: Path) -> Dict[str, int]:
    """
    Count valid images per class in the uploads folder and drop unreadable files.
    Returns e.g. {"NORMAL": 12, "PNEUMONIA": 8, "total": 20}.
    """
    counts = {c: 0 for c in CLASS_NAMES}
    for cls in CLASS_NAMES:
        cls_dir = upload_dir / cls
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() not in VALID_EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()          # raises if the file is corrupt
                counts[cls] += 1
            except Exception:
                f.unlink(missing_ok=True)  # remove unreadable file
    counts["total"] = sum(counts[c] for c in CLASS_NAMES)
    return counts


# ---------------------------------------------------------------------------
# 4. Dataset summary for the visualisations (notebook & UI)
# ---------------------------------------------------------------------------
def dataset_summary(data_dir: Path, sample_per_class: int = 60) -> Dict:
    """
    Gather lightweight statistics used by the 3+ feature visualisations:
      * class counts       (feature 1 - class balance)
      * mean pixel intensity/brightness per class (feature 2)
      * image width/height distribution (feature 3 - acquisition consistency)
    Only a sample of images is inspected so this stays fast.
    """
    summary = {"class_counts": {}, "brightness": {}, "dimensions": []}
    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
        summary["class_counts"][cls] = len(files)

        brightness_vals = []
        for f in files[:sample_per_class]:
            try:
                with Image.open(f) as im:
                    w, h = im.size
                    summary["dimensions"].append({"class": cls, "width": w, "height": h})
                    gray = np.asarray(im.convert("L"), dtype=np.float32)
                    brightness_vals.append(float(gray.mean()))
            except Exception:
                continue
        if brightness_vals:
            summary["brightness"][cls] = {
                "mean": float(np.mean(brightness_vals)),
                "std": float(np.std(brightness_vals)),
                "values": brightness_vals,
            }
    return summary
