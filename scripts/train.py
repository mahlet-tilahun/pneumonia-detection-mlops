"""
train.py — one-command offline training entrypoint.

Usage:
    python scripts/train.py                 # MobileNetV2, 15 epochs (default)
    python scripts/train.py --model cnn --epochs 20

Produces:
    models/pneumonia_model.keras   (+ .h5 copy)
    models/training_history.json
    models/evaluation_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

# allow "python scripts/train.py" from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TRAIN_DIR, TEST_DIR
from src.preprocessing import build_generators, compute_class_weights
from src.model import build_model, train_model, evaluate_model, save_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet", choices=["mobilenet", "cnn"])
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    if not any(TRAIN_DIR.glob("*/*")):
        print(f"No training images found in {TRAIN_DIR}.")
        print("Run  python scripts/download_data.py  first (see README).")
        sys.exit(1)

    print("Building data generators ...")
    train_gen, val_gen, test_gen = build_generators()
    class_weights = compute_class_weights(train_gen)
    print("Class weights:", class_weights)

    print(f"Building {args.model} model ...")
    model = build_model(model_type=args.model)
    model.summary()

    print(f"Training for up to {args.epochs} epochs (early stopping enabled) ...")
    model, _ = train_model(model, train_gen, val_gen, epochs=args.epochs,
                          class_weights=class_weights)

    print("Evaluating on the held-out test set ...")
    metrics = evaluate_model(model, test_gen)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k in ("accuracy", "precision", "recall", "f1_score", "auc", "loss")},
                     indent=2))

    save_model(model)
    print("Saved model to models/. Training complete.")


if __name__ == "__main__":
    main()
