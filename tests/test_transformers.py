"""Tests unitarios para los transformadores personalizados de Scikit-Learn."""

import numpy as np
import pandas as pd
import pytest
from src.features.custom_transformers import (
    B2BRatioFeatureGenerator,
    RobustOutlierWinsorizer,
    SafeLog1pTransformer,
)


def test_b2b_ratio_feature_generator(synthetic_churn_df: pd.DataFrame):
    transformer = B2BRatioFeatureGenerator()
    
    # Fit y Transform
    transformed = transformer.fit_transform(synthetic_churn_df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert "seat_utilization_rate" in transformed.columns
    assert "unresolved_tickets_ratio" in transformed.columns
    assert "avg_logins_per_active_user" in transformed.columns
    assert "mrr_per_contracted_seat" in transformed.columns
    
    # Comprobar límites matemáticos
    assert (transformed["seat_utilization_rate"] >= 0.0).all()
    assert (transformed["unresolved_tickets_ratio"] >= 0.0).all()
    assert (transformed["unresolved_tickets_ratio"] <= 1.0).all()


def test_b2b_ratio_feature_generator_missing_column_error():
    transformer = B2BRatioFeatureGenerator()
    df_invalid = pd.DataFrame({"col_a": [1, 2, 3]})
    with pytest.raises(ValueError):
        transformer.fit(df_invalid)


def test_robust_outlier_winsorizer(synthetic_churn_df: pd.DataFrame):
    winsorizer = RobustOutlierWinsorizer(lower_quantile=0.05, upper_quantile=0.95)
    
    # Inyectamos un outlier extremo artificialmente
    df_test = synthetic_churn_df.copy()
    df_test.loc[0, "invoice_delay_days"] = 9999.0
    
    winsorizer.fit(synthetic_churn_df)
    transformed = winsorizer.transform(df_test)
    
    # El valor recortado no debe ser 9999.0
    assert transformed.loc[0, "invoice_delay_days"] < 9999.0
    assert transformed.loc[0, "invoice_delay_days"] <= winsorizer.upper_bounds_["invoice_delay_days"]


def test_safe_log1p_transformer():
    df = pd.DataFrame({
        "skewed_a": [0.0, 10.0, 100.0, 1000.0],
        "negative_anomaly": [-5.0, 0.0, 5.0, 10.0],
    })
    transformer = SafeLog1pTransformer(columns=["skewed_a", "negative_anomaly"])
    res = transformer.fit_transform(df)
    
    # No deben generarse NaNs ni Infs
    assert not np.isnan(res.values).any()
    assert not np.isinf(res.values).any()
    assert res.loc[0, "negative_anomaly"] == 0.0  # -5 truncado a 0 -> log1p(0) = 0
