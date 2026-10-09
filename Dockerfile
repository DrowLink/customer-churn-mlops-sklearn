# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for B2B Customer Churn MLOps Microservice

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install .

# Stage 2: Production Distroless/Slim Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Security: Non-privileged dedicated system user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source code, configurations, and artifacts
COPY configs/ ./configs/
COPY src/ ./src/
COPY main.py ./

# Create artifacts and reports directories with correct ownership
RUN mkdir -p models/artifacts reports/drift data && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Zero-dependency Python healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--serve", "--host", "0.0.0.0", "--port", "8000"]
