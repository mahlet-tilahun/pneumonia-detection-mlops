"""
locustfile.py — flood-request simulation for the deployed model.

Simulates many concurrent users hammering the /predict endpoint with a real
chest X-ray image, plus lighter hits on /status and /visualizations.

Run against a locally running or cloud-deployed instance:

    # 1. make sure the API is running (locally: uvicorn app.main:app --port 8000)
    # 2. point SAMPLE_IMAGE at any real .jpeg X-ray (see README)
    # 3. start Locust:
    locust -f locustfile.py --host http://localhost:8000
    # then open http://localhost:8089 and set users / spawn-rate

    # Or fully headless (writes CSV reports used in the README results table):
    locust -f locustfile.py --host http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 2m --headless \
           --csv results/locust_1container

Record the latency / RPS for 1, 2, 3 ... Docker containers and paste the numbers
into README.md under "Flood Request Simulation".
"""
import os
from pathlib import Path
from locust import HttpUser, task, between

# A real X-ray sample to send. Defaults to the first test image if present.
ROOT = Path(__file__).resolve().parent
_DEFAULT = ROOT / "data" / "test" / "PNEUMONIA"
SAMPLE_IMAGE = os.environ.get("SAMPLE_IMAGE", "")

if not SAMPLE_IMAGE:
    imgs = list(_DEFAULT.glob("*.jpeg")) + list(_DEFAULT.glob("*.jpg")) if _DEFAULT.exists() else []
    SAMPLE_IMAGE = str(imgs[0]) if imgs else ""

_IMAGE_BYTES = Path(SAMPLE_IMAGE).read_bytes() if SAMPLE_IMAGE and Path(SAMPLE_IMAGE).exists() else None


class PneumoniaUser(HttpUser):
    wait_time = between(0.5, 2.5)

    @task(5)
    def predict(self):
        if _IMAGE_BYTES is None:
            # No sample image available — hit health instead so the test still runs.
            self.client.get("/health", name="/health (no sample image)")
            return
        files = {"file": ("xray.jpeg", _IMAGE_BYTES, "image/jpeg")}
        self.client.post("/predict", files=files, name="/predict")

    @task(2)
    def status(self):
        self.client.get("/status", name="/status")

    @task(1)
    def visualizations(self):
        self.client.get("/visualizations", name="/visualizations")
