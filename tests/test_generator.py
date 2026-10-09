"""Tests unitarios para el generador de datos sintéticos B2B."""

import pandas as pd

from src.data.generator import generate_synthetic_b2b_churn_data


def test_generate_synthetic_b2b_churn_data_shape_and_columns():
    n_samples = 200
    df = generate_synthetic_b2b_churn_data(n_samples=n_samples, random_state=123)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    assert "churn" in df.columns
    assert "customer_id" in df.columns
    assert "monthly_recurring_revenue" in df.columns

    # Comprobar que no hay duplicados en customer_id
    assert df["customer_id"].nunique() == n_samples


def test_generate_synthetic_b2b_churn_data_distributions():
    df = generate_synthetic_b2b_churn_data(
        n_samples=1000,
        churn_base_rate=0.15,
        random_state=42,
        inject_missing=False,
    )

    # El churn rate debe estar razonablemente acotado entre 10% y 25%
    churn_rate = df["churn"].mean()
    assert 0.08 <= churn_rate <= 0.25

    # Valores numéricos no deben ser negativos donde no tenga sentido
    assert (df["monthly_recurring_revenue"] >= 0).all()
    assert (df["contracted_seats"] >= 1).all()
    assert (df["active_users_last_30d"] >= 0).all()
    assert (df["contract_duration_months"] >= 1).all()


def test_generate_synthetic_b2b_churn_data_missing_injection():
    df = generate_synthetic_b2b_churn_data(
        n_samples=500,
        inject_missing=True,
        random_state=42,
    )
    # Debe contener valores nulos en NPS y storage tras inyección
    assert df["nps_score"].isna().sum() > 0
    assert df["storage_usage_pct"].isna().sum() > 0
