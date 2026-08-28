"""Unit tests for Joblib serialization, checksum verification, and inference."""

from pathlib import Path
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features.custom_transformers import B2BRatioFeatureGenerator
from src.inference.artifacts import (
    ModelArtifactMetadata,
    load_model_artifact,
    save_model_artifact,
)
from src.inference.predict import ChurnInferenceEngine, CustomerChurnInputSchema
from src.pipelines.builder import build_b2b_preprocessor, build_full_churn_pipeline


def test_artifacts_save_and_load(tmp_path: Path, synthetic_churn_df: pd.DataFrame, numeric_features, categorical_features):
    preprocessor = build_b2b_preprocessor(numeric_features, categorical_features)
    clf = HistGradientBoostingClassifier(random_state=42)
    pipeline = build_full_churn_pipeline(preprocessor, clf)
    
    X = synthetic_churn_df.drop(columns=["churn", "customer_id"])
    y = synthetic_churn_df["churn"].values
    pipeline.fit(X, y)
    
    metadata = ModelArtifactMetadata(
        model_name="test_model",
        version="v1.0.0",
        algorithm="HistGradientBoosting",
        best_threshold=0.42,
        calibrated=False,
        input_features=list(X.columns),
    )
    
    p_path, m_path = save_model_artifact(pipeline, metadata, output_dir=tmp_path)
    assert p_path.exists()
    assert m_path.exists()
    
    loaded_pipe, loaded_meta = load_model_artifact(p_path, m_path, verify_checksum=True)
    assert loaded_pipe is not None
    assert loaded_meta.best_threshold == 0.42


def test_inference_engine_single_and_batch(tmp_path: Path, synthetic_churn_df: pd.DataFrame, numeric_features, categorical_features):
    preprocessor = build_b2b_preprocessor(numeric_features, categorical_features)
    clf = HistGradientBoostingClassifier(random_state=42)
    pipeline = build_full_churn_pipeline(preprocessor, clf)
    
    X = synthetic_churn_df.drop(columns=["churn", "customer_id"])
    y = synthetic_churn_df["churn"].values
    pipeline.fit(X, y)
    
    metadata = ModelArtifactMetadata(
        model_name="churn_prod",
        version="0.1.0",
        algorithm="HistGradientBoosting",
        best_threshold=0.35,
        calibrated=False,
        input_features=list(X.columns),
    )
    p_path, m_path = save_model_artifact(pipeline, metadata, output_dir=tmp_path)
    
    engine = ChurnInferenceEngine(pipeline_path=p_path, metadata_path=m_path)
    
    # 1. Single Predict
    single_input = {
        "customer_id": "CUST-999",
        "industry": "FinTech",
        "company_size_bracket": "51-200",
        "plan_tier": "Enterprise",
        "billing_cycle": "Monthly",
        "payment_method": "Credit_Card",
        "auto_renew_enabled": False,
        "contract_duration_months": 3,
        "contracted_seats": 50,
        "active_users_last_30d": 10,
        "monthly_recurring_revenue": 3000.0,
        "avg_daily_logins": 5.0,
        "feature_adoption_score": 25.0,
        "storage_usage_pct": 80.0,
        "support_tickets_count": 8,
        "unresolved_tickets_count": 5,
        "nps_score": 3.0,
        "invoice_delay_days": 18.0,
    }
    
    single_res = engine.predict_single(single_input)
    assert single_res.customer_id == "CUST-999"
    assert 0.0 <= single_res.churn_probability <= 1.0
    assert single_res.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    # 2. Batch Predict
    df_batch = synthetic_churn_df.drop(columns=["churn"])
    batch_out_path = tmp_path / "batch_predictions.csv"
    df_res = engine.batch_predict(df_batch, output_path=batch_out_path)
    
    assert "churn_probability" in df_res.columns
    assert "churn_prediction" in df_res.columns
    assert "risk_level" in df_res.columns
    assert batch_out_path.exists()
