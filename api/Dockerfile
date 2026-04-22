# API + worker image (mock engine; no GPU / PyTorch). V2A-08.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the API needs to import (keep build context small via .dockerignore).
COPY api/requirements-docker.txt /app/api/requirements-docker.txt
RUN pip install --upgrade pip \
    && pip install -r /app/api/requirements-docker.txt

COPY src /app/src
COPY app /app/app
COPY api /app/api

EXPOSE 8000

# Health check is defined in docker-compose (api only; worker reuses this image).
# Default: API (override in compose for the worker)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
