"""
Central configuration for the Pneumonia Detection MLOps project.

Every module (preprocessing, model, prediction, API) imports from here so that
image size, paths and class names never drift out of sync.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root so the code works locally & in Docker)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"      # data/train/NORMAL , data/train/PNEUMONIA
TEST_DIR = DATA_DIR / "test"        # data/test/NORMAL  , data/test/PNEUMONIA
VAL_DIR = DATA_DIR / "val"          # optional validation split (created if present)
UPLOAD_DIR = DATA_DIR / "uploads"   # where user-uploaded retraining images land
SAMPLE_DIR = DATA_DIR / "sample"    # small sample shipped in the Docker image so the
                                    # deployed app's Data Insights charts render without
                                    # the (git-ignored) full dataset present

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "pneumonia_model.keras"   # native Keras format (primary)
MODEL_PATH_H5 = MODELS_DIR / "pneumonia_model.h5"   # legacy .h5 (rubric-compatible)
HISTORY_PATH = MODELS_DIR / "training_history.json"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# ---------------------------------------------------------------------------
# Image / model hyper-parameters
# ---------------------------------------------------------------------------
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3                         # X-rays are grayscale but MobileNet needs 3
BATCH_SIZE = 32
SEED = 42

# flow_from_directory sorts class folders alphabetically -> NORMAL=0, PNEUMONIA=1
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Retraining trigger: auto-retrain once this many new images have been uploaded
RETRAIN_TRIGGER_THRESHOLD = 20

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create the directories the app writes to (safe to call repeatedly)."""
    for d in (MODELS_DIR, UPLOAD_DIR, UPLOAD_DIR / "NORMAL", UPLOAD_DIR / "PNEUMONIA"):
        d.mkdir(parents=True, exist_ok=True)
