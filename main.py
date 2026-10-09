"""End-to-End MLOps Pipeline Orchestrator CLI for Customer Churn Prediction.

Executes the complete production lifecycle:
1. Synthetic B2B SaaS dataset generation and ingestion.
2. Feature engineering and modular pipeline assembly.
3. Comparative model training and tuning (HistGradientBoosting vs. RandomForest).
4. Probability calibration using CalibratedClassifierCV.
5. Dynamic financial decision threshold optimization.
6. Permutation Importance feature attribution.
7. Secure atomic artifact serialization with SHA-256 integrity checks.
8. Real-time (Online) and Batch inference demonstration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.data.generator import generate_synthetic_b2b_churn_data
from src.evaluation.calibration import (
    evaluate_calibration,
    find_optimal_decision_threshold,
    fit_calibrated_model,
)
from src.evaluation.explainability import (
    compute_permutation_importance,
    get_top_features_summary,
)
from src.inference.artifacts import ModelArtifactMetadata, save_model_artifact
from src.inference.predict import ChurnInferenceEngine
from src.models.train import train_and_compare_models
from src.monitoring.drift import DriftMonitor

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("churn_mlops_main")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    """Executes the complete MLOps pipeline end-to-end."""
    cfg = load_config(config_path)
    logger.info("=== STARTING END-TO-END PIPELINE: %s ===", cfg["project"]["name"])

    # 1. Data Ingestion & Generation
    raw_cfg = cfg["data"]["synthetic"]
    df = generate_synthetic_b2b_churn_data(
        n_samples=raw_cfg["n_samples"],
        churn_base_rate=raw_cfg["churn_rate"],
        random_state=cfg["project"]["random_state"],
        inject_outliers=True,
        inject_missing=True,
    )

    raw_data_path = Path(cfg["data"]["raw_data_path"])
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_data_path, index=False)
    logger.info("Raw dataset persisted to %s", raw_data_path)

    # 2. Model Training & Comparison
    num_cols = cfg["features"]["numeric_features"]
    cat_cols = cfg["features"]["categorical_features"]
    target_col = cfg["features"]["target_column"]
    biz_cfg = cfg["business_metrics"]

    train_results = train_and_compare_models(
        df=df,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        target_col=target_col,
        test_size=cfg["model_training"]["test_size"],
        random_state=cfg["project"]["random_state"],
        business_intervention_cost=biz_cfg["intervention_cost_usd"],
        business_intervention_success_rate=biz_cfg["intervention_success_rate"],
        business_ltv_multiplier=biz_cfg["ltv_mrr_multiplier"],
        use_halving_search=True,
    )

    champion_name = train_results["best_model_name"]
    champion_pipeline = train_results["champion_pipeline"]
    splits = train_results["data_splits"]

    logger.info("\n--- MODEL BENCHMARK RESULTS ---")
    for m_name, res in train_results["comparison_results"].items():
        logger.info(
            "Model: %s | Test Metrics: %s",
            m_name,
            res["metrics"],
        )

    # 3. Probability Calibration (CalibratedClassifierCV)
    logger.info("\n--- PROBABILITY CALIBRATION ---")
    calibrated_pipeline = fit_calibrated_model(
        estimator=champion_pipeline,
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        method="isotonic",
        cv=5,
    )

    test_calibrated_probs = calibrated_pipeline.predict_proba(splits["X_test"])[:, 1]
    calib_metrics = evaluate_calibration(splits["y_test"], test_calibrated_probs)

    # 4. Dynamic Business Threshold Optimization
    logger.info("\n--- DYNAMIC FINANCIAL THRESHOLD OPTIMIZATION ---")
    best_threshold, max_benefit, _ = find_optimal_decision_threshold(
        y_true=splits["y_test"],
        y_prob=test_calibrated_probs,
        mrr_series=splits["mrr_test"],
        intervention_cost=biz_cfg["intervention_cost_usd"],
        intervention_success_rate=biz_cfg["intervention_success_rate"],
        ltv_mrr_multiplier=biz_cfg["ltv_mrr_multiplier"],
    )

    # 5. Feature Attribution with Permutation Importance
    logger.info("\n--- PERMUTATION IMPORTANCE ANALYSIS ---")
    df_importance = compute_permutation_importance(
        estimator=calibrated_pipeline,
        X_eval=splits["X_test"],
        y_eval=splits["y_test"],
        scoring="roc_auc",
        n_repeats=5,
        random_state=cfg["project"]["random_state"],
    )
    logger.info("\n%s\n", get_top_features_summary(df_importance, top_n=10))

    # 6. Secure Artifact Serialization & Governance
    logger.info("\n--- ARTIFACT SERIALIZATION & GOVERNANCE ---")
    metadata = ModelArtifactMetadata(
        model_name=cfg["artifacts"]["model_name"],
        version=cfg["project"]["version"],
        algorithm=champion_name,
        best_threshold=round(best_threshold, 4),
        calibrated=True,
        input_features=num_cols + cat_cols,
        validation_metrics={
            "test_roc_auc": round(float(roc_auc_score(splits["y_test"], test_calibrated_probs)), 4),
            "test_brier_score": round(calib_metrics["brier_score"], 4),
            "test_ece": round(calib_metrics["ece"], 4),
            "max_projected_financial_benefit_usd": round(max_benefit, 2),
        },
        business_parameters=biz_cfg,
        baseline_statistics=train_results.get("baseline_summary", {}),
    )

    p_path, m_path = save_model_artifact(
        pipeline=calibrated_pipeline,
        metadata=metadata,
        output_dir=cfg["artifacts"]["model_dir"],
    )
    logger.info("Artifacts saved successfully: \n- Pipeline: %s \n- Metadata: %s", p_path, m_path)

    # 7. Production Inference Demonstration
    logger.info("\n--- PRODUCTION INFERENCE DEMONSTRATION ---")
    engine = ChurnInferenceEngine(pipeline_path=p_path, metadata_path=m_path)

    sample_customer = {
        "customer_id": "CUST-CRITICAL-007",
        "industry": "Enterprise SaaS",
        "company_size_bracket": "51-200",
        "plan_tier": "Enterprise",
        "billing_cycle": "Monthly",
        "payment_method": "Credit_Card",
        "auto_renew_enabled": False,
        "contract_duration_months": 2,
        "contracted_seats": 100,
        "active_users_last_30d": 20,
        "monthly_recurring_revenue": 7500.0,
        "avg_daily_logins": 12.0,
        "feature_adoption_score": 22.5,
        "storage_usage_pct": 92.0,
        "support_tickets_count": 14,
        "unresolved_tickets_count": 9,
        "nps_score": 2.0,
        "invoice_delay_days": 24.5,
    }

    pred_res = engine.predict_single(sample_customer)
    logger.info(
        "Real-Time Prediction for At-Risk Customer:\n%s", pred_res.model_dump_json(indent=2)
    )

    logger.info("\n=== MLOPS PIPELINE COMPLETED SUCCESSFULLY ===")


def run_drift_evaluation(
    current_data_path: str | None = None,
    config_path: str = "configs/config.yaml",
    output_report_dir: str = "reports/drift",
) -> None:
    """Evaluates data and feature drift on production payloads against training baseline."""
    cfg = load_config(config_path)
    logger.info("=== EVALUATING DATA AND FEATURE DRIFT ===")

    raw_data_path = Path(cfg["data"]["raw_data_path"])
    if not raw_data_path.exists():
        logger.info(
            "Reference raw dataset not found at %s. Generating baseline reference...", raw_data_path
        )
        ref_df = generate_synthetic_b2b_churn_data(
            n_samples=cfg["data"]["synthetic"]["n_samples"],
            churn_base_rate=cfg["data"]["synthetic"]["churn_rate"],
            random_state=cfg["project"]["random_state"],
        )
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        ref_df.to_csv(raw_data_path, index=False)
    else:
        logger.info("Loading baseline reference dataset from %s", raw_data_path)
        ref_df = pd.read_csv(raw_data_path)

    # Load current dataset to evaluate
    if current_data_path:
        curr_path = Path(current_data_path)
        logger.info("Loading current inference dataset from %s", curr_path)
        curr_df = (
            pd.read_parquet(curr_path) if curr_path.suffix == ".parquet" else pd.read_csv(curr_path)
        )
    else:
        logger.info(
            "No current input specified. Simulating shifted production batch for drift audit..."
        )
        curr_df = generate_synthetic_b2b_churn_data(
            n_samples=800,
            churn_base_rate=0.35,  # Shifted churn rate
            random_state=999,
        )
        # Inject artificial distribution shifts
        curr_df["nps_score"] = np.clip(curr_df["nps_score"] - 3.0, 0, 10)
        curr_df["invoice_delay_days"] = curr_df["invoice_delay_days"] * 2.2
        curr_df["support_tickets_count"] = curr_df["support_tickets_count"] + 5

    num_cols = cfg["features"]["numeric_features"]
    cat_cols = cfg["features"]["categorical_features"]

    monitor = DriftMonitor(
        reference_df=ref_df,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    report = monitor.evaluate_drift(curr_df)

    out_dir = Path(output_report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

    import json

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info("Drift audit completed. Report saved to: %s", report_file)
    logger.info(
        "Summary: Dataset Drifted = %s | Drifted Features = %d/%d (%s)",
        report.is_dataset_drifted,
        report.drifted_features_count,
        report.total_features_evaluated,
        ", ".join(report.drifted_features) if report.drifted_features else "None",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Churn MLOps Pipeline Runner & API Server"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI REST inference microservice with Uvicorn",
    )
    parser.add_argument(
        "--evaluate-drift",
        action="store_true",
        help="Evaluate feature and data drift against baseline dataset",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input data (CSV/Parquet) for drift evaluation or batch inference",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind API server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind API server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for local development",
    )
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        logger.info(
            "Starting FastAPI Churn Inference Microservice on http://%s:%d ...",
            args.host,
            args.port,
        )
        uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)
    elif args.evaluate_drift:
        run_drift_evaluation(current_data_path=args.input, config_path=args.config)
    else:
        run_pipeline(config_path=args.config)
