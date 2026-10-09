"""Tests para la construcción y ejecución del Pipeline completo de Scikit-Learn."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.pipelines.builder import build_b2b_preprocessor, build_full_churn_pipeline


def test_build_b2b_preprocessor(
    synthetic_churn_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
):
    preprocessor = build_b2b_preprocessor(
        numeric_features=numeric_features,
        categorical_ohe_features=categorical_features,
    )

    X = synthetic_churn_df.drop(columns=["churn", "customer_id"])
    preprocessed_matrix = preprocessor.fit_transform(X)

    assert isinstance(preprocessed_matrix, np.ndarray)
    assert preprocessed_matrix.ndim == 2
    assert preprocessed_matrix.shape[0] == len(synthetic_churn_df)
    assert not np.isnan(preprocessed_matrix).any()


def test_build_full_churn_pipeline_predict(
    synthetic_churn_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
):
    preprocessor = build_b2b_preprocessor(
        numeric_features=numeric_features,
        categorical_ohe_features=categorical_features,
    )
    clf = HistGradientBoostingClassifier(random_state=42)
    pipeline = build_full_churn_pipeline(preprocessor=preprocessor, classifier=clf)

    X = synthetic_churn_df.drop(columns=["churn", "customer_id"])
    y = synthetic_churn_df["churn"].values

    pipeline.fit(X, y)

    # Probar predicciones
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
