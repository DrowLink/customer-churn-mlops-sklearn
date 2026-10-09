"""ColumnTransformer and Scikit-Learn Production Pipeline Assembler.

Builds a leak-free preprocessor that:
1. Calculates domain ratio features (`B2BRatioFeatureGenerator`).
2. Robustly winsorizes numeric outliers (`RobustOutlierWinsorizer`).
3. Preprocesses numeric features: Median imputation + Scalers (`RobustScaler` / `StandardScaler`).
4. Encodes categorical features:
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for standard categories.
   - `TargetEncoder(cv=5, smooth='auto')` for high-cardinality features.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder

from src.features.custom_transformers import (
    B2BRatioFeatureGenerator,
    RobustOutlierWinsorizer,
    SafeLog1pTransformer,
)

logger = logging.getLogger(__name__)


def build_b2b_preprocessor(
    numeric_features: Sequence[str],
    categorical_ohe_features: Sequence[str],
    categorical_target_encode_features: Sequence[str] | None = None,
    apply_winsorization: bool = True,
    apply_log_transform: bool = True,
    log_features: Sequence[str] | None = None,
    scaler_type: str = "robust",
) -> Pipeline:
    """Builds a scikit-learn preprocessing Pipeline with modular ColumnTransformer.

    Args:
        numeric_features (Sequence[str]): Base numerical column names.
        categorical_ohe_features (Sequence[str]): Categorical columns to one-hot encode.
        categorical_target_encode_features (Sequence[str] | None): Columns to target encode.
        apply_winsorization (bool): If True, clips numeric outliers to [1%, 99%] quantiles.
        apply_log_transform (bool): If True, applies log1p to skewed columns.
        log_features (Sequence[str] | None): Columns to transform with log1p.
        scaler_type (str): 'robust' for RobustScaler or 'standard' for StandardScaler.

    Returns:
        Pipeline: Preprocessing pipeline ready for `.fit()` and `.transform()`.
    """
    logger.info("Building Preprocessing and Feature Engineering Pipeline...")

    if categorical_target_encode_features is None:
        categorical_target_encode_features = []

    if log_features is None:
        log_features = [
            "monthly_recurring_revenue",
            "contract_duration_months",
            "support_tickets_count",
        ]

    # 1. Numeric Pipeline
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if apply_winsorization:
        num_steps.append(
            ("winsorizer", RobustOutlierWinsorizer(lower_quantile=0.01, upper_quantile=0.99))
        )

    if apply_log_transform:
        num_steps.append(("log1p", SafeLog1pTransformer(columns=log_features)))

    if scaler_type == "robust":
        num_steps.append(("scaler", RobustScaler()))
    else:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    # 2. Categorical Pipeline (OneHotEncoder ignoring unseen test categories)
    ohe_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
            ),
        ]
    )

    # 3. High-cardinality Target Encoder if provided
    if categorical_target_encode_features:
        target_encoder_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "target_enc",
                    TargetEncoder(
                        categories="auto",
                        target_type="binary",
                        smooth="auto",
                        cv=5,
                        random_state=42,
                    ),
                ),
            ]
        )

    # 4. Assembled ColumnTransformer
    derived_ratio_names = [
        "seat_utilization_rate",
        "unresolved_tickets_ratio",
        "avg_logins_per_active_user",
        "mrr_per_contracted_seat",
    ]
    all_numeric = list(numeric_features) + derived_ratio_names

    final_transformers: list[tuple[str, Any, Sequence[str]]] = [
        ("num", numeric_transformer, all_numeric),
        ("cat_ohe", ohe_transformer, list(categorical_ohe_features)),
    ]
    if categorical_target_encode_features:
        final_transformers.append(
            ("cat_target_enc", target_encoder_transformer, list(categorical_target_encode_features))
        )

    final_column_preprocessor = ColumnTransformer(
        transformers=final_transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    full_feature_pipeline = Pipeline(
        steps=[
            ("ratio_generator", B2BRatioFeatureGenerator(drop_intermediates=False)),
            ("preprocessor", final_column_preprocessor),
        ]
    )

    return full_feature_pipeline


def build_full_churn_pipeline(
    preprocessor: Pipeline,
    classifier: Any,
) -> Pipeline:
    """Combines preprocessor and estimator into an end-to-end scikit-learn Pipeline.

    Args:
        preprocessor (Pipeline): Feature preprocessing pipeline.
        classifier (Any): Estimator classifier (e.g. HistGradientBoostingClassifier).

    Returns:
        Pipeline: End-to-end pipeline ready for fit/predict.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
