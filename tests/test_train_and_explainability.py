"""Tests unitarios para entrenamiento comparativo y explicabilidad (Permutation Importance)."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.explainability import compute_permutation_importance, get_top_features_summary
from src.models.train import train_and_compare_models


def test_compute_permutation_importance_and_summary(synthetic_churn_df, numeric_features):
    """Verifica el cálculo de Permutation Importance y el formato del reporte resumen."""
    X = synthetic_churn_df[numeric_features].fillna(0)
    y = synthetic_churn_df["churn"].values

    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X, y)

    df_importance = compute_permutation_importance(
        estimator=clf,
        X_eval=X,
        y_eval=y,
        scoring="roc_auc",
        n_repeats=2,
        random_state=42,
    )

    assert isinstance(df_importance, pd.DataFrame)
    assert "feature" in df_importance.columns
    assert "importance_mean" in df_importance.columns
    assert len(df_importance) == len(numeric_features)

    summary_text = get_top_features_summary(df_importance, top_n=5)
    assert "Rank" in summary_text
    assert "Feature" in summary_text


def test_train_and_compare_models_fast(synthetic_churn_df, numeric_features, categorical_features):
    """Prueba el flujo de entrenamiento comparativo con búsqueda básica."""
    # Usamos submuestra pequeña para velocidad en test suite
    sample_df = synthetic_churn_df.head(200).copy()

    results = train_and_compare_models(
        df=sample_df,
        numeric_features=numeric_features[:3],
        categorical_features=categorical_features[:2],
        target_col="churn",
        test_size=0.25,
        random_state=42,
        use_halving_search=False,
    )

    assert "best_model_name" in results
    assert "champion_pipeline" in results
    assert "comparison_results" in results
    assert "baseline_summary" in results
    assert results["best_model_name"] in ["hist_gradient_boosting", "random_forest"]
