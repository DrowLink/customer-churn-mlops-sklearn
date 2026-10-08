# Technical Architecture & Implementation Plan: Production MLOps Hardening & FastAPI Serving

**Feature ID**: `001-production-mlops-hardening`  
**Status**: `Approved / In Progress`  

---

## 1. System Architecture Overview

```mermaid
graph TD
    subgraph Client Applications
        CRM[Salesforce / CRM Webhook]
        BatchScheduler[Airflow / Cron Job]
        DevUser[Data Scientist / Dev CLI]
    end

    subgraph Serving & API Layer (FastAPI)
        Router[FastAPI App: /v1/predict, /v1/predict/batch, /health]
        Validator[Pydantic Runtime Schema Validator]
        DriftDetector[Drift & Anomaly Monitor (PSI / KS-Test)]
    end

    subgraph Inference & Pipeline Engine
        Engine[ChurnInferenceEngine]
        ArtifactGov[Checksum Verifier & Metadata Loader]
        SKPipeline[Scikit-Learn Calibrated Pipeline]
    end

    subgraph Storage & Observability
        ArtifactsDir[models/artifacts/*.joblib + metadata.json]
        LogsTelemetry[Structured JSON Logs & Metrics]
        DriftReports[reports/drift/*.json]
    end

    CRM -->|POST /v1/predict| Router
    BatchScheduler -->|POST /v1/predict/batch| Router
    DevUser -->|python main.py --serve| Router

    Router --> Validator
    Validator --> DriftDetector
    DriftDetector --> Engine
    Engine --> ArtifactGov
    ArtifactGov --> ArtifactsDir
    Engine --> SKPipeline
    Engine --> LogsTelemetry
    DriftDetector --> DriftReports
```

---

## 2. Technical Decisions & Trade-Offs

| Decision Area | Selected Technology | Alternative Considered | Rationale & Trade-offs |
| :--- | :--- | :--- | :--- |
| **REST Serving Framework** | **FastAPI + Uvicorn** | Flask, TorchServe, BentoML | FastAPI provides native async support, automated OpenAPI docs, and instant integration with Pydantic v2 schemas already present in the codebase. |
| **Data Drift Detection** | **Lightweight PSI + KS-test module** | Heavy Evidently AI / Alibi-Detect suite | Keeps dependencies lean, eliminates bloat, and provides zero-overhead calculations directly within standard `scipy` and `numpy`. |
| **Containerization** | **Multi-stage Dockerfile** | Single-stage heavy container | Separates build dependencies from slim production runtime (`python:3.11-slim`), cutting image footprint by >60%. |
| **Continuous Integration** | **GitHub Actions** | GitLab CI, Jenkins | Standard GitHub repository workflow, free tier runners, cross-platform validation (Linux/Windows). |
| **Developer Ergonomics** | **Makefile & PowerShell script** | Pure manual python invocations | Single-command execution: `make test`, `make lint`, `make train`, `make serve`. |

---

## 3. Module & File Architecture

```
customer-churn-mlops-sklearn/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated CI: lint, type-check, pytest, coverage
├── .specify/                           # Spec Kit (Spec-Driven Development)
│   ├── memory/constitution.md
│   ├── templates/
│   │   ├── spec-template.md
│   │   ├── plan-template.md
│   │   └── tasks-template.md
│   └── specify.yaml
├── specs/                              # Living Feature Specifications
│   └── 001-production-mlops-hardening/
│       ├── spec.md                     # Business requirements and acceptance criteria
│       ├── plan.md                     # This architectural document
│       └── tasks.md                    # Executable checklist of tasks
├── src/
│   ├── api/                            # NEW: FastAPI microservice
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI application instance & lifespan
│   │   ├── routes.py                   # /health, /predict, /predict/batch endpoints
│   │   └── schemas.py                  # Pydantic request/response schemas
│   ├── monitoring/                     # NEW: Data & drift monitoring
│   │   ├── __init__.py
│   │   └── drift.py                    # PSI, Kolmogorov-Smirnov, baseline summary
│   ├── data/
│   ├── features/
│   ├── pipelines/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── tests/
│   ├── test_api.py                     # NEW: FastAPI test suite (TestClient)
│   ├── test_drift.py                   # NEW: Drift detection test suite
│   ├── ... (existing unit tests)
├── Dockerfile                          # NEW: Multi-stage containerization
├── docker-compose.yml                  # NEW: Local dev orchestration
├── Makefile                            # NEW: Developer commands
├── main.py                             # Enhanced CLI orchestrator (--serve, --drift)
└── pyproject.toml                      # Updated dependencies (fastapi, uvicorn, httpx)
```

---

## 4. Testing, Validation & Verification Strategy

1. **Unit Testing (`pytest tests/`)**:
   - Test each FastAPI route with valid and invalid payloads.
   - Verify proper error codes (`422` on schema violation, `503` if model not found).
   - Test drift detection with identical distributions (PSI ~ 0) and shifted distributions (PSI > 0.3).
2. **Integration & Integrity Testing**:
   - Verify startup integrity verification (corrupting model checksum must raise fatal error).
   - Test batch prediction throughput and memory efficiency.
3. **CI Pipeline Quality Gates**:
   - Ruff linting (`ruff check .` and `ruff format --check .`).
   - Pytest execution with coverage reporting (`pytest --cov=src --cov-fail-under=85`).
