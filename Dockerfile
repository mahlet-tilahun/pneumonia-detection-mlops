# Pneumonia Detection MLOps — production image
# Python 3.11 (TensorFlow 2.15 does NOT support 3.13, hence the pinned base image)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# System libs required by Pillow / TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install SLIM runtime deps only (see requirements-docker.txt).
# Each step is its own layer so a network retry doesn't redo the whole install.
# High --retries/--timeout make the large TensorFlow download resilient to
# flaky connections (the wheel is ~475 MB).
ENV PIP_RETRIES=15 PIP_TIMEOUT=300 PIP_NO_CACHE_DIR=1
COPY requirements-docker.txt .
RUN pip install --upgrade pip
RUN pip install tensorflow==2.15.0
RUN pip install -r requirements-docker.txt

# Copy the project
COPY . .

# Cloud platforms inject $PORT; default to 8000 locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
