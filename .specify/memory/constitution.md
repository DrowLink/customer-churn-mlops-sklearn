# Project Constitution: Customer Churn MLOps (B2B SaaS)

## 1. Purpose & Identity
This project delivers a production-grade, business-optimized Machine Learning Operations (MLOps) system for early B2B SaaS customer churn prediction. Every architectural decision, model evaluation, and deployment workflow must directly prioritize business net financial ROI (balancing Customer Lifetime Value vs. proactive Customer Success intervention cost) while enforcing strict software engineering and data science rigor.

---

## 2. Core Architectural Principles

### 2.1 Data Leakage Zero Tolerance
- **Strict Separation**: Preprocessing, imputation, feature scaling, winsorization, and target encoding must learn statistics exclusively on training folds (`fit` / `fit_transform`) and apply them blindly on validation/holdout sets (`transform`).
- **Scikit-Learn Standard API**: All custom feature transforms must inherit from `BaseEstimator` and `TransformerMixin` and adhere to immutable contracts with unit test coverage.
- **Cross-Validation Integrity**: Cross-validation splits must preserve business group boundaries or stratification according to SaaS subscription lifecycle.

### 2.2 Financial & Business-Centric Optimization
- **Asymmetric Loss Modeling**: In B2B SaaS, False Negatives (losing an enterprise account) are orders of magnitude more expensive than False Positives (cost of a Customer Success proactive call).
- **Custom Utility Function**: Hyperparameter search and decision thresholding must optimize the Net Financial Benefit metric:
  $$\text{Net Benefit} = \sum_{i \in TP} (\text{SuccessRate} \times \text{LTV}_i - \text{InterventionCost}) - \sum_{j \in FP} \text{InterventionCost} - \sum_{k \in FN} \text{LTV}_k$$
- **Calibrated Probabilities**: Raw classifier probabilities must be calibrated via `CalibratedClassifierCV` (Isotonic regression or Sigmoid) and validated using Brier Score and Expected Calibration Error (ECE) before applying dynamic decision thresholds.

### 2.3 Artifact Governance & Security
- **Atomic Serialization**: Models and pipelines are serialized with joblib accompanied by cryptographic SHA-256 checksums to guarantee artifact immutability.
- **Strict Type Validation**: All inference inputs and metadata schemas must be validated using Pydantic models with explicit constraints, boundary checks, and UTC timestamping.
- **Decoupled Pipeline Architecture**: Preprocessing and inference pipelines must be self-contained and reproducible without relying on ambient global state.

---

## 3. Engineering & Code Quality Standards

### 3.1 Code Style & Linting
- **Python Version**: Python 3.10+ compatibility.
- **Linter & Formatter**: Strict adherence to Ruff (`line-length = 100`, rules: `E`, `F`, `I`, `N`, `W`, `UP`, `B`, `A`, `C4`, `T20`).
- **Typing**: Type hints are mandatory across all public functions, classes, and pipeline interfaces (`from __future__ import annotations`).

### 3.2 Testing Standards
- **Coverage**: Core business scorers, custom transformers, generator logic, and inference engines must maintain >= 90% test coverage.
- **Test Categories**:
  - Unit tests for each custom transformer (checking edge cases: zeroes, NaNs, unseen categories, extreme outliers).
  - Integration tests for end-to-end pipeline fit, serialization, deserialization, and inference matching.
  - Regression and schema validation tests for prediction payloads.

### 3.3 Documentation & Transparency
- **Model Explainability**: Global feature attribution must rely on holdout Permutation Importance rather than biased tree-based Mean Decrease in Impurity (MDI).
- **Auditability**: Every generated model artifact must store full execution metadata (git commit SHA, python environment versions, validation metrics, threshold parameters, training duration).

---

## 4. Operational Boundaries & Tech Stack
- **Core Framework**: `scikit-learn`, `pandas`, `numpy`, `scipy`, `pydantic`, `joblib`, `pyyaml`.
- **Packaging & Environment**: Modern `pyproject.toml` configuration.
- **Inference Latency Budget**:
  - Single customer real-time inference: `< 50 ms` (p95).
  - Batch scoring: Vectorized execution capable of handling 50,000 customers in `< 30 s`.
