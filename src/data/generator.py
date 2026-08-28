"""Realistic synthetic B2B SaaS customer data generator with churn signals.

This module simulates metrics for Software-as-a-Service (B2B SaaS) customer accounts, incorporating:
- Product engagement metrics (feature adoption, login frequency, storage utilization).
- Service & support friction (ticket volume, unresolved tickets, NPS).
- Financial & contractual metrics (MRR, contract duration, invoice delay days, payment methods).
- Latent non-linear relationships and interactions driving customer churn.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_b2b_churn_data(
    n_samples: int = 10000,
    churn_base_rate: float = 0.15,
    random_state: int = 42,
    inject_outliers: bool = True,
    inject_missing: bool = True,
) -> pd.DataFrame:
    """Generates a realistic synthetic dataset of B2B SaaS customer accounts.

    Args:
        n_samples (int): Number of customer records to generate. Defaults to 10,000.
        churn_base_rate (float): Expected base population churn rate.
        random_state (int): Seed for statistical reproducibility.
        inject_outliers (bool): If True, introduces plausible domain outliers (e.g., extreme delay).
        inject_missing (bool): If True, introduces realistic missing values in optional fields (e.g., NPS).

    Returns:
        pd.DataFrame: DataFrame containing feature columns and the binary `churn` target (0 or 1).
    """
    rng = np.random.default_rng(random_state)
    logger.info("Generating %d synthetic B2B customer records with seed=%d...", n_samples, random_state)

    # 1. Identifiers and Firmographics
    customer_ids = [f"CUST-{100000 + i}" for i in range(n_samples)]
    
    industries = rng.choice(
        ["FinTech", "HealthTech", "E-commerce", "EdTech", "Enterprise SaaS", "Logistics"],
        size=n_samples,
        p=[0.20, 0.15, 0.25, 0.15, 0.15, 0.10],
    )
    
    company_size_bracket = rng.choice(
        ["1-10", "11-50", "51-200", "201-1000", "1000+"],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.10, 0.05],
    )
    
    # Tier and Seat assignments conditioned on company size
    plan_tier = []
    contracted_seats = []
    mrr_list = []
    
    for size in company_size_bracket:
        if size == "1-10":
            tier = rng.choice(["Starter", "Professional"], p=[0.7, 0.3])
            seats = rng.integers(1, 10)
            base_seat_price = 25.0 if tier == "Starter" else 50.0
        elif size == "11-50":
            tier = rng.choice(["Starter", "Professional", "Enterprise"], p=[0.2, 0.6, 0.2])
            seats = rng.integers(11, 50)
            base_seat_price = 45.0
        elif size == "51-200":
            tier = rng.choice(["Professional", "Enterprise"], p=[0.4, 0.6])
            seats = rng.integers(51, 200)
            base_seat_price = 60.0
        elif size == "201-1000":
            tier = "Enterprise"
            seats = rng.integers(201, 1000)
            base_seat_price = 75.0
        else:  # 1000+
            tier = "Enterprise"
            seats = rng.integers(1000, 5000)
            base_seat_price = 70.0
            
        mrr = float(seats * base_seat_price * rng.uniform(0.85, 1.15))
        plan_tier.append(tier)
        contracted_seats.append(seats)
        mrr_list.append(round(mrr, 2))

    billing_cycle = rng.choice(
        ["Monthly", "Annual", "Multi-Year"],
        size=n_samples,
        p=[0.55, 0.35, 0.10],
    )
    
    payment_method = rng.choice(
        ["Credit_Card", "Bank_Transfer", "ACH"],
        size=n_samples,
        p=[0.50, 0.30, 0.20],
    )
    
    auto_renew_enabled = rng.choice(
        [True, False],
        size=n_samples,
        p=[0.75, 0.25],
    )

    contract_duration_months = np.clip(
        rng.exponential(scale=18.0, size=n_samples) + 1,
        1,
        72,
    ).astype(int)

    # 2. Product Engagement and Usage Metrics
    utilization_mean = rng.uniform(0.3, 0.95, size=n_samples)
    active_users = np.maximum(
        1,
        np.round(np.array(contracted_seats) * utilization_mean).astype(int)
    )
    
    avg_daily_logins = np.round(
        active_users * rng.uniform(0.5, 3.5, size=n_samples) + rng.normal(0, 1, size=n_samples),
        2
    )
    avg_daily_logins = np.maximum(0.1, avg_daily_logins)

    feature_adoption_score = np.clip(
        rng.normal(loc=65, scale=20, size=n_samples),
        5.0,
        100.0,
    )

    storage_usage_pct = np.clip(
        rng.beta(a=2, b=3, size=n_samples) * 100,
        1.0,
        100.0,
    )

    # 3. Support, Friction and Customer Satisfaction
    support_tickets_count = rng.poisson(lam=3.0, size=n_samples)
    unresolved_ratio = rng.beta(a=1, b=4, size=n_samples)
    unresolved_tickets_count = np.round(support_tickets_count * unresolved_ratio).astype(int)
    
    nps_score = np.clip(
        rng.normal(loc=7.5, scale=2.5, size=n_samples),
        0.0,
        10.0,
    ).round(1)

    # Billing delay days
    invoice_delay_days = np.clip(
        rng.exponential(scale=4.0, size=n_samples) - 1.5,
        0.0,
        60.0,
    ).round(1)

    # 4. Latent Causal Churn Mechanism (Log-Odds formulation)
    utilization_rate = active_users / np.array(contracted_seats)
    
    # Baseline log-odds for ~15% churn rate
    log_odds = -2.8
    
    # Risk factors (increase churn log-odds):
    log_odds += (1.0 - utilization_rate) * 2.2                # Low seat utilization
    log_odds += (invoice_delay_days / 15.0) * 1.5             # Payment delays
    log_odds += (unresolved_tickets_count / (support_tickets_count + 1)) * 2.0  # Support friction
    log_odds += np.where(billing_cycle == "Monthly", 0.6, -0.4) # Monthly contracts are more volatile
    log_odds += np.where(~auto_renew_enabled, 0.8, -0.3)      # No auto-renew
    log_odds += np.where(nps_score <= 5.0, 1.4, -0.5)         # Detractor NPS
    log_odds += np.where(feature_adoption_score < 40.0, 1.2, -0.4) # Low feature adoption

    # Protective factors (decrease churn log-odds):
    log_odds += np.where(plan_tier == "Enterprise", -0.7, 0.1) # Enterprise stickiness
    log_odds += -(contract_duration_months / 36.0) * 1.0       # Mature accounts are more loyal
    
    # Stochastic noise
    log_odds += rng.normal(0, 0.4, size=n_samples)
    
    # Sigmoid link function: P(Churn=1) = 1 / (1 + exp(-log_odds))
    prob_churn = 1.0 / (1.0 + np.exp(-log_odds))
    churn = rng.binomial(n=1, p=prob_churn, size=n_samples)

    # Build DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "industry": industries,
        "company_size_bracket": company_size_bracket,
        "plan_tier": plan_tier,
        "billing_cycle": billing_cycle,
        "payment_method": payment_method,
        "auto_renew_enabled": auto_renew_enabled,
        "contract_duration_months": contract_duration_months,
        "contracted_seats": contracted_seats,
        "active_users_last_30d": active_users,
        "monthly_recurring_revenue": mrr_list,
        "avg_daily_logins": avg_daily_logins,
        "feature_adoption_score": feature_adoption_score,
        "storage_usage_pct": storage_usage_pct,
        "support_tickets_count": support_tickets_count,
        "unresolved_tickets_count": unresolved_tickets_count,
        "nps_score": nps_score,
        "invoice_delay_days": invoice_delay_days,
        "churn": churn,
    })

    # Outlier and missing data injection for robust testing
    if inject_outliers:
        outlier_indices = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        df.loc[outlier_indices, "invoice_delay_days"] = rng.uniform(70, 180, size=len(outlier_indices))
        
        ticket_outliers = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        df.loc[ticket_outliers, "support_tickets_count"] = rng.integers(30, 80, size=len(ticket_outliers))

    if inject_missing:
        missing_nps_indices = rng.choice(n_samples, size=int(n_samples * 0.12), replace=False)
        df.loc[missing_nps_indices, "nps_score"] = np.nan
        
        missing_storage = rng.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        df.loc[missing_storage, "storage_usage_pct"] = np.nan

    logger.info("Dataset generated successfully. Observed Churn Rate: %.2f%%", df["churn"].mean() * 100)
    return df
