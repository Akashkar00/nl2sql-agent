# Multi-stage Dockerfile for NL2SQL FastAPI service.
# CPU image. For GPU serving, swap base image + add CUDA deps.

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for sqlite + faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        sqlite3 \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Skip torch CUDA in this image — install CPU torch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/

EXPOSE 8000

# Note: model loading happens on first request. First /query call will be slow.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
