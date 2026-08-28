"""Model Explainability and Feature Importance Module.

Implements model-agnostic `permutation_importance` on holdout validation/test sets to avoid
the known Gini impurity (MDI) bias towards continuous or high-cardinality features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def compute_permutation_importance(
    estimator: Any,
    X_eval: pd.DataFrame | np.ndarray,
    y_eval: np.ndarray,
    scoring: str | Any = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Computes permutation importance for all input features.

    Args:
        estimator (Any): Fitted model or pipeline.
        X_eval (pd.DataFrame | np.ndarray): Evaluation dataset (avoid using Train).
        y_eval (np.ndarray): Ground truth labels.
        scoring (str | Any): Scoring metric to measure performance drop ('roc_auc', 'f1', etc.).
        n_repeats (int): Number of permutations per feature.
        random_state (int): Random seed.
        n_jobs (int): CPU parallelism.

    Returns:
        pd.DataFrame: DataFrame sorted by `importance_mean` descending with means and standard deviations.
    """
    logger.info(
        "Computing Permutation Importance (n_repeats=%d, scoring='%s')...",
        n_repeats,
        str(scoring),
    )

    perm_result = permutation_importance(
        estimator=estimator,
        X=X_eval,
        y=y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if isinstance(X_eval, pd.DataFrame):
        feature_names = list(X_eval.columns)
    elif hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)
    else:
        feature_names = [f"feature_{i}" for i in range(X_eval.shape[1])]

    df_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_importance


def get_top_features_summary(df_importance: pd.DataFrame, top_n: int = 10) -> str:
    """Generates a formatted text summary of top churn driving features."""
    top_df = df_importance.head(top_n)
    lines = [f"{'Rank':<5} | {'Feature':<30} | {'Importance (Mean ± Std)':<25}"]
    lines.append("-" * 65)
    for idx, row in top_df.iterrows():
        lines.append(
            f"{idx + 1:<5} | {row['feature']:<30} | {row['importance_mean']:>8.4f} ± {row['importance_std']:<8.4f}"
        )
    return "\n".join(lines)
