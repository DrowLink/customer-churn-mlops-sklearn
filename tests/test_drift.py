"""Tests unitarios y de integración para el módulo de Drift Monitoring."""

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    DriftMonitor,
    compute_categorical_psi,
    compute_ks_test,
    compute_psi,
    extract_baseline_summary,
)


def test_compute_psi_identical_distributions():
    """Valida que dos distribuciones idénticas tengan un PSI cercano a 0 (< 0.05)."""
    np.random.seed(42)
    ref = np.random.normal(loc=50.0, scale=10.0, size=2000)
    curr = np.random.normal(loc=50.0, scale=10.0, size=2000)

    psi_val = compute_psi(ref, curr, num_buckets=10)
    assert psi_val < 0.05, f"PSI esperado < 0.05 para distribuciones iguales, obtenido: {psi_val}"


def test_compute_psi_shifted_distributions():
    """Valida que una distribución con desplazamiento severo tenga un PSI > 0.25."""
    np.random.seed(42)
    ref = np.random.normal(loc=50.0, scale=10.0, size=2000)
    # Desplazamiento significativo en media y varianza
    curr = np.random.normal(loc=75.0, scale=15.0, size=2000)

    psi_val = compute_psi(ref, curr, num_buckets=10)
    assert psi_val > 0.25, f"PSI esperado > 0.25 para distribuciones alteradas, obtenido: {psi_val}"


def test_compute_ks_test():
    """Verifica el test de Kolmogorov-Smirnov para datos idénticos vs alterados."""
    np.random.seed(42)
    ref = np.random.normal(loc=100.0, scale=15.0, size=1000)
    same_dist = np.random.normal(loc=100.0, scale=15.0, size=1000)
    shifted_dist = np.random.normal(loc=120.0, scale=15.0, size=1000)

    stat_same, pval_same = compute_ks_test(ref, same_dist)
    stat_shift, pval_shift = compute_ks_test(ref, shifted_dist)

    assert pval_same > 0.01
    assert pval_shift < 0.001
    assert stat_shift > stat_same


def test_compute_categorical_psi():
    """Prueba el cálculo de PSI para variables categóricas."""
    ref = pd.Series(["Basic"] * 500 + ["Pro"] * 300 + ["Enterprise"] * 200)
    curr_same = pd.Series(["Basic"] * 500 + ["Pro"] * 300 + ["Enterprise"] * 200)
    curr_shifted = pd.Series(["Basic"] * 100 + ["Pro"] * 100 + ["Enterprise"] * 800)

    psi_same = compute_categorical_psi(ref, curr_same)
    psi_shifted = compute_categorical_psi(ref, curr_shifted)

    assert psi_same < 0.05
    assert psi_shifted > 0.25


def test_extract_baseline_summary(synthetic_churn_df, numeric_features, categorical_features):
    """Verifica que el resumen estadístico base contenga las dimensiones esperadas."""
    summary = extract_baseline_summary(
        df=synthetic_churn_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    assert summary["n_samples"] == len(synthetic_churn_df)
    assert "numeric_features" in summary
    assert "categorical_features" in summary
    for num_col in numeric_features:
        assert num_col in summary["numeric_features"]
        assert "mean" in summary["numeric_features"][num_col]
        assert "quantiles" in summary["numeric_features"][num_col]


def test_drift_monitor_evaluation(synthetic_churn_df, numeric_features, categorical_features):
    """Evalúa el DriftMonitor en escenarios sin drift y con drift severo."""
    monitor = DriftMonitor(
        reference_df=synthetic_churn_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # 1. Dataset estable
    stable_batch = synthetic_churn_df.sample(n=200, random_state=123).copy()
    stable_report = monitor.evaluate_drift(stable_batch)

    assert not stable_report.is_dataset_drifted
    assert stable_report.drifted_features_count == 0

    # 2. Dataset con drift inducido
    drifted_batch = synthetic_churn_df.sample(n=200, random_state=456).copy()
    drifted_batch["nps_score"] = 0.0
    drifted_batch["monthly_recurring_revenue"] = drifted_batch["monthly_recurring_revenue"] * 5.0
    drifted_batch["plan_tier"] = "Enterprise"

    drifted_report = monitor.evaluate_drift(drifted_batch)

    assert drifted_report.is_dataset_drifted
    assert drifted_report.drifted_features_count >= 2
    assert "nps_score" in drifted_report.drifted_features

    # Validar serialización de reporte
    dict_rep = drifted_report.to_dict()
    assert "feature_metrics" in dict_rep
    assert dict_rep["is_dataset_drifted"] is True
