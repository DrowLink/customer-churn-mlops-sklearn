# Customer Churn MLOps with Scikit-Learn (B2B SaaS)

Production-grade Machine Learning and MLOps system in Python for early Churn prediction in B2B SaaS platforms. Optimized financially using custom business metrics (MRR vs. Retention Intervention Cost), calibrated probabilities, leak-free Scikit-Learn pipelines, and artifact governance.

---

## 🏛️ System Architecture

```
customer-churn-mlops-sklearn/
├── configs/
│   └── config.yaml                     # Feature definitions, HPO spaces, and financial parameters
├── src/
│   ├── data/
│   │   └── generator.py                # Realistic B2B synthetic data generator (engagement, billing, MRR)
│   ├── features/
│   │   └── custom_transformers.py      # Safe transformers (BaseEstimator/TransformerMixin)
│   ├── pipelines/
│   │   └── builder.py                  # ColumnTransformer & Pipeline assembler
│   ├── models/
│   │   ├── train.py                    # Comparative HPO training (HalvingRandomSearchCV / StratifiedKFold)
│   │   └── metrics.py                  # Custom Business Scorer (make_scorer, Cost vs MRR)
│   ├── evaluation/
│   │   ├── calibration.py              # CalibratedClassifierCV & Dynamic Threshold Optimization
│   │   └── explainability.py           # Model-agnostic Permutation Importance
│   └── inference/
│       ├── artifacts.py                # Joblib serialization, SHA-256 Checksums, and Pydantic metadata
│       └── predict.py                  # Real-time Online (Pydantic) & Batch (Parquet/CSV) inference engine
├── tests/                              # Comprehensive unit test suite with pytest
│   ├── conftest.py
│   ├── test_generator.py
│   ├── test_transformers.py
│   ├── test_pipeline.py
│   ├── test_metrics.py
│   └── test_inference.py
├── pyproject.toml                      # Package specifications, dependencies, and linters (ruff)
├── main.py                             # End-to-end CLI orchestrator
└── README.md
```

---

## 💡 Technical and Architectural Highlights

### 1. Realistic B2B Synthetic Data Generation
- Generates customer data reflecting true B2B SaaS dynamics with non-linear interactions:
  - **Product Engagement**: License utilization rate (`active_users / contracted_seats`), logins per active seat, and feature adoption score.
  - **Support Friction**: Support ticket volume, unresolved ticket ratio, and NPS score.
  - **Contract & Financial Health**: Invoice delay days, contract term (monthly vs. multi-year), and MRR.
  - **Causal Mechanism**: Latent log-odds mapped via sigmoid function to determine ground-truth churn probability.

### 2. Leak-Free Preprocessing (`BaseEstimator` & `TransformerMixin`)
- Custom transformer classes adhere strictly to Scikit-Learn's API contracts:
  - `B2BRatioFeatureGenerator`: Derives key domain ratios without division-by-zero errors.
  - `RobustOutlierWinsorizer`: Learns quantile cutoffs [1%, 99%] exclusively in `.fit()` (Train) and applies them in `.transform()`.
  - `SafeLog1pTransformer`: Safely applies $\log(1+x)$ to skewed distributions with non-negative bounds.
- Modular `ColumnTransformer` combines `OneHotEncoder(handle_unknown='ignore')`, `TargetEncoder(cv=5, smooth='auto')`, and robust scalers.

### 3. Business-Driven Optimization (`make_scorer`)
- In B2B SaaS, a False Negative (losing a high-MRR Enterprise customer) costs significantly more than a False Positive (a $150 proactive Customer Success call).
- We formulate the **Net Financial Benefit** function:
$$\text{Benefit} = \sum_{i \in TP} (\text{SuccessRate} \times \text{LTV}_i - \text{InterventionCost}) - \sum_{j \in FP} \text{InterventionCost} - \sum_{k \in FN} \text{LTV}_k$$
- Wrapped into Scikit-Learn via `make_scorer` to directly guide hyperparameter search (`HalvingRandomSearchCV`).

### 4. Probability Calibration & Dynamic Thresholding
- Tree models with `class_weight='balanced'` produce shifted, uncalibrated probabilities.
- We utilize `CalibratedClassifierCV(method='isotonic', cv=5)` to output well-calibrated probabilities, verified with **Brier Score** and **Expected Calibration Error (ECE)**.
- Replaces static 0.5 decision thresholds with **Dynamic Threshold Optimization**, maximizing ROI along the financial curve.

### 5. Explainability with Permutation Importance
- Avoids Gini impurity (MDI) bias present in Random Forest, which inflates the importance of high-cardinality/continuous features.
- Computes permutation importance on holdout validation data.

### 6. MLOps, Serialization, and Artifact Governance
- **Atomic Serialization**: Joblib serialization paired with SHA-256 checksums to guarantee artifact integrity.
- **Pydantic Schemas**: Strict schema validation for online inference payloads and structured metadata tracking (UTC timestamps, validation metrics, business parameters).
- **Inference Engine**: Dual support for single-entity real-time scoring and vectorized high-throughput batch scoring.

---

## 🚀 Quickstart & Usage

### 1. Prerequisites
Python 3.10 or higher.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-org/customer-churn-mlops-sklearn.git
cd customer-churn-mlops-sklearn

# Install package and development dependencies
pip install -e ".[dev]"
```

### 3. Run the Unit Test Suite
```bash
python -m pytest tests/ -v
```

### 4. Execute the End-to-End Pipeline
```bash
python main.py --config configs/config.yaml
```

### 5. Launch the Production FastAPI Serving Microservice
```bash
python main.py --serve --port 8000
```
Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
- **Healthcheck**: `http://127.0.0.1:8000/health`

---

## 🌐 REST API Serving Endpoints

### Single Account Scoring (`POST /v1/predict`)
```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-ENT-FIN-9921",
    "industry": "FinTech",
    "company_size_bracket": "51-200",
    "plan_tier": "Enterprise",
    "billing_cycle": "Monthly",
    "payment_method": "Bank_Transfer",
    "auto_renew_enabled": false,
    "contract_duration_months": 5,
    "contracted_seats": 100,
    "active_users_last_30d": 18,
    "monthly_recurring_revenue": 6250.0,
    "avg_daily_logins": 7.0,
    "feature_adoption_score": 24.5,
    "storage_usage_pct": 72.0,
    "support_tickets_count": 11,
    "unresolved_tickets_count": 7,
    "nps_score": 2.5,
    "invoice_delay_days": 19.0
  }'
```

### Vectorized Batch Scoring (`POST /v1/predict/batch`)
```bash
curl -X POST http://127.0.0.1:8000/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [...],
    "sort_by_risk": true
  }'
```

---

## 📋 Spec-Driven Development (Spec Kit)

This repository follows the **Spec Kit** standard for version-controlled, spec-driven architecture:
- **Constitution**: [constitution.md](.specify/memory/constitution.md) (Architecture boundaries, zero data leakage, and business ROI rules)
- **Living Specifications**: [specs/](specs/) (Features, plans, acceptance criteria, and task checklists)

---

## 🔍 Python SDK Ingestion Example

```python
from src.inference.predict import ChurnInferenceEngine

# Load the inference engine with the latest calibrated model
engine = ChurnInferenceEngine(
    pipeline_path="models/artifacts/b2b_churn_pipeline_latest.joblib",
    metadata_path="models/artifacts/b2b_churn_pipeline_latest_metadata.json"
)

# Example customer payload
customer = {
    "customer_id": "CUST-ENTERPRISE-01",
    "industry": "FinTech",
    "company_size_bracket": "51-200",
    "plan_tier": "Enterprise",
    "billing_cycle": "Monthly",
    "payment_method": "Bank_Transfer",
    "auto_renew_enabled": False,
    "contract_duration_months": 4,
    "contracted_seats": 80,
    "active_users_last_30d": 15,
    "monthly_recurring_revenue": 5200.0,
    "avg_daily_logins": 10.0,
    "feature_adoption_score": 30.0,
    "storage_usage_pct": 75.0,
    "support_tickets_count": 9,
    "unresolved_tickets_count": 6,
    "nps_score": 3.0,
    "invoice_delay_days": 21.0,
}

result = engine.predict_single(customer)
print(result.model_dump_json(indent=2))
```

