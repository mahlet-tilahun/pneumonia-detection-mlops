# Pneumonia Detection — End-to-End MLOps Pipeline

A full machine-learning lifecycle project that classifies **chest X-ray images** as
**NORMAL** or **PNEUMONIA**, deployed as a Dockerised web app with prediction, data
visualisations, bulk data upload, and one-click model retraining.

> **Non-tabular extension:** this project uses real medical **image** data, building on the
> tabular [Stroke Risk Prediction](https://github.com/mahlet-tilahun/stroke-risk-prediction-ml) summative.

---

## Table of Contents
1. [Demo Video & Live App](#demo-video--live-app)
2. [Project Description](#project-description)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Setup — Run Locally](#setup--run-locally)
6. [Train the Model](#train-the-model)
7. [Run the Notebook](#run-the-notebook)
8. [Run the Web App / API](#run-the-web-app--api)
9. [Flood Request Simulation (Locust + Docker)](#flood-request-simulation-locust--docker)
10. [API Reference](#api-reference)
11. [Model Performance](#model-performance)

---

## Demo Video & Live App
**Demo video (YouTube):** `PASTE_YOUR_YOUTUBE_LINK_HERE`

**Live app:** https://pneumonia-detection-mlops.onrender.com

---

## Project Description
The system detects pneumonia from chest X-rays using **MobileNetV2 transfer learning**.
It demonstrates the complete ML pipeline required by the rubric:

- **Data acquisition** — Kaggle Chest X-Ray dataset (`scripts/download_data.py`)
- **Data processing** — resize, RGB conversion, rescaling, augmentation, class weighting (`src/preprocessing.py`)
- **Model creation & testing** — CNN / MobileNetV2 with regularisation + early stopping (`src/model.py`)
- **Prediction** — single-image inference (`src/prediction.py`)
- **Retraining + trigger** — upload new data and retrain via a button, or auto-trigger on data threshold
- **API** — FastAPI (`app/main.py`)
- **UI** — web dashboard: prediction, uptime monitor, 3 interpreted visualisations, upload & retrain
- **Cloud deployment** — Docker + Render.com
- **Load testing** — Locust flood simulation across multiple Docker containers

---

## Dataset
**Chest X-Ray Images (Pneumonia)** — Paul Mooney, Kaggle
<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

5,863 real X-ray JPEGs · 2 classes (NORMAL / PNEUMONIA) · pre-split into train/test/val.

### Option A — Automatic download (recommended)
1. Create a free [Kaggle](https://www.kaggle.com) account.
2. Kaggle → *Account* → **Create New API Token** → downloads `kaggle.json`.
3. Place it at `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Mac/Linux).
4. Run:
   ```bash
   pip install kagglehub
   python scripts/download_data.py
   ```

### Option B — Manual
Download & unzip from the Kaggle link above, then copy the `train/`, `test/`, `val/`
folders (each containing `NORMAL/` and `PNEUMONIA/`) into this project's `data/` folder.

---

## Project Structure
```
pneumonia-detection-mlops/
├── README.md
├── requirements.txt
├── Dockerfile  · docker-compose.yml · nginx.conf · render.yaml
├── locustfile.py
├── notebook/
│   └── pneumonia_detection.ipynb        # full ML lifecycle notebook
├── src/
│   ├── config.py
│   ├── preprocessing.py                 # data loading & image preprocessing
│   ├── model.py                         # build / train / evaluate / retrain
│   └── prediction.py                    # single-image inference
├── app/
│   ├── main.py                          # FastAPI (API + serves the UI)
│   └── templates/index.html             # web dashboard
├── scripts/
│   ├── download_data.py                 # dataset acquisition
│   └── train.py                         # one-command training
├── tests/test_api.py                    # smoke tests (pytest)
├── data/  (train/ test/ val/ uploads/)  # populated by download_data.py
└── models/                              # pneumonia_model.keras / .h5 + metrics
```

---

## Setup — Run Locally

> **Use Python 3.10 or 3.11.** TensorFlow 2.15 does **not** support Python 3.13.
> Check with `python --version`. If you have 3.13, install 3.11 from python.org and use it below.

```bash
# from the project folder
py -3.11 -m venv .venv                 # Windows (or: python3.11 -m venv .venv)
.venv\Scripts\activate                 # Windows PowerShell:  .venv\Scripts\Activate.ps1
#   macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

---

## Train the Model
```bash
python scripts/download_data.py            # 1. get the data
python scripts/train.py                    # 2. train (MobileNetV2, ~15 epochs)
#   options:  python scripts/train.py --model cnn --epochs 20
```
This writes `models/pneumonia_model.keras` (+ `.h5`), `training_history.json` and
`evaluation_metrics.json`. On CPU, training takes a while — lower `--epochs` for a quick run.

---

## Run the Notebook
```bash
jupyter notebook notebook/pneumonia_detection.ipynb
```
The notebook reproduces every step: preprocessing, the 3 interpreted visualisations, training,
the full evaluation metric suite, prediction, saving, and the retraining function.

---

## Run the Web App / API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open <http://localhost:8000>. The dashboard has three tabs:
- **Predict** — upload one X-ray, get NORMAL/PNEUMONIA + confidence
- **Data Insights** — 3 interpreted feature charts + live model metrics
- **Upload & Retrain** — bulk-upload labelled images and trigger retraining

A live **status bar** shows server uptime, model status, and prediction count.

---

## Flood Request Simulation (Locust + Docker)
Demonstrates how the model responds under load with **different numbers of Docker containers**.

```bash
# scale to N containers behind the nginx load balancer
docker compose up --build --scale api=1 -d      # then 2, then 3

# run the flood (headless, saves CSV report)
locust -f locustfile.py --host http://localhost:8080 \
       --users 15 --spawn-rate 5 --run-time 60s --headless \
       --csv results/locust_1container
```

### Flood Simulation Results
**Setup:** 15 concurrent users, 60 s per run, identical load at each scale. Each API
container is **capped at 1 CPU** (`docker-compose.yml`) so that scaling 1 → 2 → 3
containers adds real parallel cores on the 4-core host — making the effect measurable.

All requests (`/predict` + `/status` + `/visualizations`), aggregated:

| Containers | Requests | Failures | Median (ms) | p95 (ms) | Avg (ms) | RPS |
|-----------:|---------:|---------:|------------:|---------:|---------:|----:|
| 1 | 86  | 0 | 4,200 | 32,000 | 6,527 | 1.48 |
| 2 | 188 | 0 | 660   | 22,000 | 2,869 | 3.15 |
| 3 | 238 | 0 | 610   | 13,000 | 1,911 | 4.01 |

`/predict` endpoint only (the CPU-bound model inference):

| Containers | Requests | Failures | Median (ms) | p95 (ms) | Avg (ms) | RPS |
|-----------:|---------:|---------:|------------:|---------:|---------:|----:|
| 1 | 58  | 0 | 7,000 | 10,000 | 6,124 | 1.00 |
| 2 | 119 | 0 | 730   | 1,900  | 869   | 1.99 |
| 3 | 169 | 0 | 670   | 1,600  | 748   | 2.85 |

**Observation:** With a single CPU-bound container, 15 concurrent users saturate it —
`/predict` median latency climbs to **7 s** and throughput plateaus at **~1 req/s**.
Adding replicas scales throughput **near-linearly with cores** (predict RPS 1.00 → 1.99 →
2.85; total requests handled 86 → 188 → 238) and collapses median latency **~10×**
(7,000 ms → 670 ms). **Zero failed requests** at every scale, confirming the service stays
stable under load. This is the expected horizontal-scaling benefit: more containers = more
parallel inference capacity = lower latency and higher request throughput.

---

## API Reference
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET  | `/` | Web dashboard |
| GET  | `/health` | Liveness probe |
| GET  | `/status` | Uptime, model status, counters |
| POST | `/predict` | Predict one image (`file` = image) |
| GET  | `/visualizations` | Dataset summary for the charts |
| GET  | `/metrics` | Deployed model's evaluation metrics |
| POST | `/upload` | Bulk upload images (`files[]`, `label`) |
| POST | `/retrain` | Trigger retraining (`model_type`, `epochs`) |
| GET  | `/retrain/status` | Retraining progress |

Example prediction with `curl`:
```bash
curl -X POST http://localhost:8000/predict -F "file=@data/test/PNEUMONIA/person1_virus_6.jpeg"
```

---

## Model Performance
Actual results (MobileNetV2 transfer learning) on the held-out **624-image test set**
(from `models/evaluation_metrics.json`):

| Metric | Score |
|--------|------:|
| Accuracy | 0.846 |
| Precision | 0.885 |
| Recall (sensitivity) | 0.867 |
| F1 score | 0.876 |
| AUC | 0.929 |
| Loss | 0.334 |

**Confusion matrix** (rows = actual, cols = predicted): `[[190, 44], [52, 338]]`
(NORMAL: 190 correct, 44 misclassified; PNEUMONIA: 338 correct, 52 missed).

*(Exact numbers vary slightly between training runs due to augmentation randomness; the values
above match the committed model and the notebook's saved outputs, and appear on the app's
Data Insights tab.)*

---

## Tests
```bash
pytest -q          # smoke tests for the API + preprocessing (no GPU/dataset needed)
```

## Tech Stack
TensorFlow/Keras · FastAPI · Uvicorn · Pillow · scikit-learn · Chart.js · Docker · nginx · Locust · Render.com

## Author
Mahlet Tilahun — African Leadership University
