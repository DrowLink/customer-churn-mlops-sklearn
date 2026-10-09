"""Model Training, Comparison, and Hyperparameter Optimization (HPO).

Compares:
1. `HistGradientBoostingClassifier(class_weight='balanced')`
2. `RandomForestClassifier(class_weight='balanced')`

Applies hyperparameter optimization with `HalvingRandomSearchCV` / `StratifiedKFold`
evaluating ROC-AUC and net business return.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from src.models.metrics import b2b_churn_net_financial_benefit, create_business_scorer
from src.monitoring.drift import extract_baseline_summary
from src.pipelines.builder import build_b2b_preprocessor, build_full_churn_pipeline

logger = logging.getLogger(__name__)


def train_and_compare_models(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str = "churn",
    test_size: float = 0.20,
    random_state: int = 42,
    business_intervention_cost: float = 150.0,
    business_intervention_success_rate: float = 0.65,
    business_ltv_multiplier: float = 12.0,
    use_halving_search: bool = True,
) -> dict[str, Any]:
    """Trains, tunes hyperparameters, and benchmarks HistGradientBoosting vs RandomForest.

    Args:
        df (pd.DataFrame): Dataset with features and target column.
        numeric_features (list[str]): Numerical column names.
        categorical_features (list[str]): Categorical column names.
        target_col (str): Binary target label column.
        test_size (float): Holdout test partition ratio.
        random_state (int): Seed for reproducibility.
        business_intervention_cost (float): Unit retention intervention cost.
        business_intervention_success_rate (float): Retention success probability.
        business_ltv_multiplier (float): Multiplier for MRR to LTV estimation.
        use_halving_search (bool): If True, utilizes HalvingRandomSearchCV.

    Returns:
        dict[str, Any]: Benchmark results, champion pipeline, and data splits.
    """
    logger.info("Initializing Stratified Train/Test split (test_size=%.2f)...", test_size)
    
    X = df.drop(columns=[target_col, "customer_id"], errors="ignore")
    y = df[target_col].values
    mrr_series = df["monthly_recurring_revenue"].values

    X_train, X_test, y_train, y_test, mrr_train, mrr_test = train_test_split(
        X,
        y,
        mrr_series,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    avg_ltv = float(np.mean(mrr_train) * business_ltv_multiplier)
    business_scorer = create_business_scorer(
        intervention_cost=business_intervention_cost,
        intervention_success_rate=business_intervention_success_rate,
        average_mrr_ltv=avg_ltv,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 1. Candidate Estimator Configurations
    models_config = {
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_distributions": {
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "classifier__max_iter": [100, 150, 200],
                "classifier__max_leaf_nodes": [15, 31, 63],
                "classifier__min_samples_leaf": [10, 20, 50],
                "classifier__l2_regularization": [0.0, 0.5, 1.0, 2.0],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_distributions": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [8, 12, 16, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
    }

    comparison_results: dict[str, Any] = {}

    for model_name, cfg in models_config.items():
        logger.info("Training and tuning candidate model: %s...", model_name)
        
        preprocessor = build_b2b_preprocessor(
            numeric_features=numeric_features,
            categorical_ohe_features=categorical_features,
        )
        pipeline = build_full_churn_pipeline(
            preprocessor=preprocessor,
            classifier=cfg["estimator"],
        )

        if use_halving_search:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                search = HalvingRandomSearchCV(
                    estimator=pipeline,
                    param_distributions=cfg["param_distributions"],
                    cv=skf,
                    scoring="roc_auc",
                    factor=2,
                    resource="n_samples",
                    min_resources=1000,
                    max_resources="auto",
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0,
                )
                search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}

        # Evaluate on Test Holdout
        y_test_probs = best_model.predict_proba(X_test)[:, 1]
        y_test_preds = (y_test_probs >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_test_probs))
        pr_auc = float(average_precision_score(y_test, y_test_probs))
        f1 = float(f1_score(y_test, y_test_preds, zero_division=0))
        
        # Test financial return
        net_financial_benefit = float(
            b2b_churn_net_financial_benefit(
                y_true=y_test,
                y_pred=y_test_preds,
                sample_weight=mrr_test * business_ltv_multiplier,
                intervention_cost=business_intervention_cost,
                intervention_success_rate=business_intervention_success_rate,
            )
        )

        comparison_results[model_name] = {
            "best_pipeline": best_model,
            "best_params": best_params,
            "metrics": {
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "f1_score_th05": round(f1, 4),
                "net_financial_benefit_usd": round(net_financial_benefit, 2),
            },
            "y_test_probs": y_test_probs,
        }

        logger.info(
            "[%s] Test ROC-AUC: %.4f | PR-AUC: %.4f | Net Benefit: $%.2f",
            model_name,
            roc_auc,
            pr_auc,
            net_financial_benefit,
        )

    # Select champion based on highest test ROC-AUC / Financial Return
    best_model_name = max(
        comparison_results.keys(),
        key=lambda k: comparison_results[k]["metrics"]["roc_auc"],
    )
    logger.info(">> Selected Champion Model: %s", best_model_name)

    baseline_summary = extract_baseline_summary(
        df=X_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    return {
        "best_model_name": best_model_name,
        "champion_pipeline": comparison_results[best_model_name]["best_pipeline"],
        "comparison_results": comparison_results,
        "baseline_summary": baseline_summary,
        "data_splits": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "mrr_train": mrr_train,
            "mrr_test": mrr_test,
        },
    }
