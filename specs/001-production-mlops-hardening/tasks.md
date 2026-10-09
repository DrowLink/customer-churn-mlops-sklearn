# Implementation Tasks: Production MLOps Hardening & FastAPI Serving

**Feature ID**: `001-production-mlops-hardening`  
**Tracking**: Spec-Driven Development (SDD)  

---

## Task Progress Overview

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Dependency & Spec Kit Infrastructure | `COMPLETED` |
| **Phase 2** | Production FastAPI Serving Layer | `COMPLETED` |
| **Phase 3** | Data & Feature Drift Monitoring | `COMPLETED` |
| **Phase 4** | CI/CD, Docker & Developer Ergonomics | `COMPLETED` |
| **Phase 5** | End-to-End Testing & Verification | `COMPLETED` |

---

## Detailed Task Breakdown

### Phase 1: Dependency & Spec Kit Setup
- [x] **Task 1.1**: Initialize `.specify` directory structure (`constitution.md`, templates, and `specify.yaml`).
  - *Files*: `.specify/memory/constitution.md`, `.specify/templates/*`, `.specify/specify.yaml`
  - *Verification*: Validate file existence and Spec Kit compliance.
- [x] **Task 1.2**: Update `pyproject.toml` with optional dependencies for API serving (`fastapi`, `uvicorn`, `httpx`).
  - *Files*: `pyproject.toml`
  - *Verification*: Dependency resolution without conflicts.

### Phase 2: Production FastAPI Serving Layer
- [x] **Task 2.1**: Implement Pydantic API schemas with strict constraints and OpenAPI examples.
  - *Files*: `src/api/schemas.py`
  - *Verification*: Type validation tests on boundary inputs.
- [x] **Task 2.2**: Implement FastAPI routes (`/health`, `/v1/predict`, `/v1/predict/batch`).
  - *Files*: `src/api/routes.py`, `src/api/app.py`
  - *Verification*: Verify routes using Starlette `TestClient`.
- [x] **Task 2.3**: Integrate SHA-256 model checksum verification in FastAPI startup lifespan.
  - *Files*: `src/api/app.py`
  - *Verification*: Verify server refusal to launch if checksum fails.

### Phase 3: Data & Feature Drift Monitoring
- [x] **Task 3.1**: Create `DriftMonitor` module calculating PSI (Population Stability Index) and Kolmogorov-Smirnov statistics.
  - *Files*: `src/monitoring/drift.py`
  - *Verification*: Unit tests verifying PSI < 0.1 for identical distributions and PSI > 0.25 for shifted data.
- [x] **Task 3.2**: Add baseline summary extraction to training pipeline so reference statistics are stored alongside model metadata.
  - *Files*: `src/models/train.py`, `main.py`
  - *Verification*: Baseline feature distributions saved in artifact metadata.

### Phase 4: CI/CD, Containerization & Developer Ergonomics
- [x] **Task 4.1**: Create multi-stage `Dockerfile` and `docker-compose.yml` for serving and training.
  - *Files*: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  - *Verification*: Docker linting and clean build.
- [x] **Task 4.2**: Create GitHub Actions CI workflow for automated linting, test suite, and coverage.
  - *Files*: `.github/workflows/ci.yml`
  - *Verification*: Action syntax verification.
- [x] **Task 4.3**: Create `Makefile` and `run.ps1` for standardized developer workflows (`make test`, `make lint`, `make serve`).
  - *Files*: `Makefile`, `run.ps1`
  - *Verification*: Execute commands locally.

### Phase 5: Testing, Hardening & Final Documentation
- [x] **Task 5.1**: Implement comprehensive tests for API endpoints (`tests/test_api.py`) and drift monitoring (`tests/test_drift.py`).
  - *Files*: `tests/test_api.py`, `tests/test_drift.py`, `tests/test_train_and_explainability.py`
  - *Verification*: `pytest` runs and passes 100% of tests with >= 90% coverage.
- [x] **Task 5.2**: Update root `README.md` with API usage instructions, Docker instructions, and Spec Kit documentation.
  - *Files*: `README.md`
  - *Verification*: Markdown links and instructions validated.
