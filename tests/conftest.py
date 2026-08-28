"""Fixtures comunes y datasets para tests unitarios y de integración."""

import pytest
import pandas as pd
import numpy as np
from src.data.generator import generate_synthetic_b2b_churn_data


@pytest.fixture(scope="session")
def synthetic_churn_df() -> pd.DataFrame:
    """Fixture que genera un dataset sintético pequeño para tests rápidos."""
    return generate_synthetic_b2b_churn_data(
        n_samples=500,
        churn_base_rate=0.15,
        random_state=42,
        inject_outliers=True,
        inject_missing=True,
    )


@pytest.fixture(scope="session")
def numeric_features() -> list[str]:
    return [
        "monthly_recurring_revenue",
        "contract_duration_months",
        "contracted_seats",
        "active_users_last_30d",
        "avg_daily_logins",
        "feature_adoption_score",
        "support_tickets_count",
        "unresolved_tickets_count",
        "nps_score",
        "invoice_delay_days",
        "storage_usage_pct",
    ]


@pytest.fixture(scope="session")
def categorical_features() -> list[str]:
    return [
        "plan_tier",
        "industry",
        "company_size_bracket",
        "billing_cycle",
        "payment_method",
        "auto_renew_enabled",
    ]
