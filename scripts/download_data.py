"""
download_data.py
================
Downloads the *real* Chest X-Ray Pneumonia dataset (Paul Mooney, Kaggle) and
arranges it into the project's data/ folder as:

    data/train/NORMAL , data/train/PNEUMONIA
    data/test/NORMAL  , data/test/PNEUMONIA
    data/val/NORMAL   , data/val/PNEUMONIA   (small validation split)

Two ways to obtain the data — pick ONE:

A) Automatic (recommended) — needs a free Kaggle account + API token:
     pip install kagglehub
     python scripts/download_data.py
   The first run may ask you to place kaggle.json (see README, "Dataset" section).

B) Manual:
     1. Download from
        https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
     2. Unzip it. Inside you'll find chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}
     3. Copy those train/ test/ val/ folders into this project's data/ folder.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _copy_split(src_root: Path):
    """Copy train/test/val (with NORMAL/PNEUMONIA subfolders) into data/."""
    moved = 0
    for split in ("train", "test", "val"):
        src = src_root / split
        if not src.exists():
            continue
        for cls in ("NORMAL", "PNEUMONIA"):
            src_cls = src / cls
            if not src_cls.exists():
                continue
            dst_cls = DATA / split / cls
            dst_cls.mkdir(parents=True, exist_ok=True)
            for f in src_cls.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_cls / f.name)
                    moved += 1
    return moved


def main():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run:  pip install kagglehub")
        print("...or follow the MANUAL instructions in this file's docstring.")
        sys.exit(1)

    print("Downloading paultimothymooney/chest-xray-pneumonia via kagglehub ...")
    path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    print(f"Downloaded to cache: {path}")

    # The archive nests everything under a 'chest_xray' folder (sometimes twice).
    candidates = [path, path / "chest_xray", path / "chest_xray" / "chest_xray"]
    src_root = next((c for c in candidates if (c / "train").exists()), None)
    if src_root is None:
        print("Could not locate the train/ folder in the download. Inspect:", path)
        sys.exit(1)

    moved = _copy_split(src_root)
    print(f"Done. Copied {moved} images into {DATA}")
    for split in ("train", "test", "val"):
        for cls in ("NORMAL", "PNEUMONIA"):
            d = DATA / split / cls
            n = len(list(d.glob("*"))) if d.exists() else 0
            print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    main()
