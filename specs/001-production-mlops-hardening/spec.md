# Feature Specification: Production MLOps Hardening & FastAPI Serving

**Feature ID**: `001-production-mlops-hardening`  
**Status**: `Draft / Ready for Review`  
**Author**: MLOps Team  
**Date**: 2026-10-08  

---

## 1. Context & Business Value

### 1.1 Problem Statement
The current repository establishes a sound machine learning pipeline (B2B data generation, leak-free Scikit-Learn transformers, probability calibration, and financial net benefit scoring). However, it remains a purely local script (`main.py` CLI). It cannot be consumed in real time by customer-facing platforms (HubSpot, Salesforce, internal CRM), lacks automated continuous integration (CI/CD), offers no containerized deployment, lacks runtime data validation gates, and provides no drift monitoring to detect when model performance degrades over time.

### 1.2 Target Stakeholders
- **Customer Success (CS) Team**: Requires automated churn alerts with customer IDs, calibrated churn probabilities, and financial risk rankings to prioritize high-MRR outreach.
- **Product & Data Platform Engineers**: Require a resilient, low-latency REST API (`< 50ms` response) with healthchecks and OpenAPI contracts.
- **MLOps Engineers**: Require CI/CD verification, reproducible Docker environments, data drift tracking, and automated model governance.

### 1.3 Key Success Metrics (OKRs)
- **API Availability & Performance**: 99.9% uptime, p95 inference latency `< 40ms` for single customer scoring.
- **CI/CD & Code Quality**: 100% automated test pass rate with `>= 90%` code coverage on PRs, zero Ruff lint errors.
- **Data Quality & Drift Defense**: Automatic detection of schema anomalies prior to model scoring; proactive alerts when feature drift (PSI > 0.25) occurs.
- **Financial Impact**: Target retention ROI improvement of >= 15% through calibrated risk scoring and prioritized intervention.

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Real-time Customer Churn Scoring via REST API
> **As an** Integration Engineer for Salesforce/HubSpot,  
> **I want to** submit a customer's usage and billing payload to a REST API endpoint,  
> **So that** Customer Success managers receive instantaneous churn risk probabilities and dynamic action flags.

- **Scenario 1.1 (Valid Enterprise Customer Payload)**:
  - *Given* an active customer record containing valid usage, billing, and NPS figures.
  - *When* sent via `POST /v1/predict`.
  - *Then* return HTTP `200 OK` with JSON response containing `churn_probability`, `predicted_churn_binary`, `recommended_action`, `intervention_priority`, and `model_version`.
- **Scenario 1.2 (Invalid / Corrupted Payload)**:
  - *Given* a payload missing required fields (e.g. `monthly_recurring_revenue`) or containing out-of-range values (e.g. `nps_score = 15`).
  - *When* sent via `POST /v1/predict`.
  - *Then* return HTTP `422 Unprocessable Entity` with explicit field-level validation errors from Pydantic.

### User Story 2: Batch Inference for Entire Customer Base
> **As a** Data Pipeline Orchestrator (e.g., Airflow / Prefect),  
> **I want to** submit a batch of 10,000+ customer records via CSV or Parquet,  
> **So that** high-risk accounts are flagged in bulk nightly.

- **Scenario 2.1 (Nightly Batch Scoring)**:
  - *Given* a batch dataset of accounts.
  - *When* executed via `POST /v1/predict/batch` or CLI `python main.py --mode batch --input data/batch.csv`.
  - *Then* return scored records sorted descending by Net Financial Risk in `< 10s`.

### User Story 3: Data Drift & Anomaly Gatekeeper
> **As an** MLOps Engineer,  
> **I want to** validate inference distributions against the baseline training distribution,  
> **So that** we detect data drift before bad predictions damage customer retention.

- **Scenario 3.1 (Severe Feature Drift Detection)**:
  - *Given* a sudden shift in customer metrics (e.g., widespread login drop during platform outage).
  - *When* the drift detector computes Population Stability Index (PSI) or Kolmogorov-Smirnov tests.
  - *Then* flag drift warnings and generate a drift summary report.

---

## 3. Scope & Requirements

### 3.1 Functional Requirements
- **FR-01 (FastAPI Service)**: Expose `/health`, `/v1/predict`, `/v1/predict/batch`, and `/docs`.
- **FR-02 (Data Validation Layer)**: Provide runtime validation schemas checking bounds, types, and required SaaS dimensions before feeding Scikit-Learn transformers.
- **FR-03 (Drift Monitoring Module)**: Compute drift metrics (PSI, KS-test, Wasserstein distance) comparing production inference payloads against the training baseline.
- **FR-04 (CLI & Orchestrator Enhancement)**: Support flags in `main.py` for `--train`, `--serve`, `--evaluate-drift`, and `--batch-predict`.

### 3.2 Non-Functional Requirements
- **NFR-01 (Reproducibility & Packaging)**: Multi-stage Docker container (`python:3.11-slim`) minimizing image size and eliminating host environment discrepancies.
- **NFR-02 (Automation & CI)**: GitHub Actions workflow executing Ruff linting, formatting check, and pytest suite on Ubuntu and Windows.
- **NFR-03 (Security & Integrity)**: Verify model SHA-256 hash at server startup; abort startup immediately if checksum does not match metadata.

### 3.3 Out of Scope for Phase 1
- Full Kubernetes Helm chart (deferred to Phase 2).
- Real-time Kafka streaming consumer (deferred to Phase 2).

---

## 4. Data & Interface Contracts

### 4.1 REST API Request Schema (`CustomerInferencePayload`)
```json
{
  "customer_id": "CUST-ENT-9921",
  "industry": "FinTech",
  "company_size_bracket": "51-200",
  "plan_tier": "Enterprise",
  "billing_cycle": "Monthly",
  "payment_method": "Bank_Transfer",
  "auto_renew_enabled": false,
  "contract_duration_months": 6,
  "contracted_seats": 100,
  "active_users_last_30d": 22,
  "monthly_recurring_revenue": 6500.0,
  "avg_daily_logins": 8.5,
  "feature_adoption_score": 28.0,
  "storage_usage_pct": 82.0,
  "support_tickets_count": 12,
  "unresolved_tickets_count": 8,
  "nps_score": 2.0,
  "invoice_delay_days": 18.0
}
```

### 4.2 REST API Response Schema (`ChurnPredictionResponse`)
```json
{
  "customer_id": "CUST-ENT-9921",
  "churn_probability": 0.8142,
  "calibrated": true,
  "decision_threshold": 0.35,
  "predicted_churn": true,
  "risk_category": "Critical Risk",
  "financial_impact": {
    "mrr": 6500.0,
    "expected_loss_usd": 78000.0,
    "recommended_intervention_budget_usd": 150.0
  },
  "top_risk_factors": [
    "High unresolved ticket ratio (66.7%)",
    "Severe license underutilization (22%)",
    "Frequent invoice payment delays (18 days)"
  ],
  "model_version": "0.1.0",
  "inference_timestamp_utc": "2026-10-08T03:30:00Z"
}
```

---

## 5. Risks & Mitigation
| Risk | Severity | Likelihood | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **Silent Data Drift** | High | Medium | Implement baseline drift monitor with automated threshold alerts (PSI > 0.2). |
| **Model Deserialization Tampering** | Critical | Low | Validate SHA-256 hash against metadata before unpickling. |
| **High Latency under Concurrent Load** | Medium | Low | Use FastAPI asynchronous endpoints and lightweight vectorized Scikit-Learn transformers. |
