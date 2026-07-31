# syntax=docker/dockerfile:1.7
# Multi-stage image for the Aegis API and arq worker.
# Same image, different CMD:
#   api:    uvicorn aegis.main:app --host 0.0.0.0 --port 8000
#   worker: arq aegis.worker.WorkerSettings

FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY migrations ./migrations

RUN pip install --upgrade pip \
    && pip install --prefix=/install .


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin aegis

COPY --from=builder /install /install
COPY --chown=aegis:aegis src ./src
COPY --chown=aegis:aegis migrations ./migrations
COPY --chown=aegis:aegis pyproject.toml ./

USER aegis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

CMD ["uvicorn", "aegis.main:app", "--host", "0.0.0.0", "--port", "8000"]
