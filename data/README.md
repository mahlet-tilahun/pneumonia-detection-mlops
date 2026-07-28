# data/

The raw dataset is **not** committed (it is ~1.2 GB). Populate this folder before training.

Expected layout after running `python scripts/download_data.py`:

```
data/
├── train/
│   ├── NORMAL/       (~1,341 images)
│   └── PNEUMONIA/    (~3,875 images)
├── test/
│   ├── NORMAL/       (~234 images)
│   └── PNEUMONIA/    (~390 images)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── uploads/          (created at runtime for user-uploaded retraining data)
    ├── NORMAL/
    └── PNEUMONIA/
```

**Dataset:** Chest X-Ray Images (Pneumonia) — Paul Mooney, Kaggle
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

See the project `README.md` → *Dataset* section for the two ways to download it.
