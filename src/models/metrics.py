"""Custom Business Cost-Benefit Metric and Scorer for B2B Churn Prediction.

In B2B SaaS, classification errors have asymmetric financial costs:
- False Negative (FN): Missing a churning customer leads to the total loss of customer LTV.
- False Positive (FP): Wastes Customer Success intervention cost ($150).
- True Positive (TP): Recovers LTV (scaled by success rate) minus intervention cost.
- True Negative (TN): Zero additional cost.

This module provides the Net Financial Benefit metric and exposes a Scikit-Learn scorer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)


def b2b_churn_net_financial_benefit(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    sample_weight: np.ndarray | None = None,
    intervention_cost: float = 150.0,
    intervention_success_rate: float = 0.65,
    average_mrr_ltv: float = 12.0 * 850.0,
) -> float:
    """Calculates the net financial return (in USD) from model-guided retention campaigns.

    Formula:
    Net Benefit = Σ [ TP * (success_rate * LTV_i - intervention_cost) ]
                - Σ [ FP * intervention_cost ]
                - Σ [ FN * LTV_i ]

    Args:
        y_true (np.ndarray | list[int]): Ground truth binary labels (0 or 1).
        y_pred (np.ndarray | list[int]): Model binary predictions based on decision threshold.
        sample_weight (np.ndarray | None): Individual customer LTV values.
        intervention_cost (float): Fixed cost per proactive retention campaign.
        intervention_success_rate (float): Success probability of retention.
        average_mrr_ltv (float): Fallback average LTV when sample_weight is not provided.

    Returns:
        float: Cumulative net financial benefit in USD (higher is better).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if sample_weight is None:
        ltv_values = np.full_like(y_true_arr, fill_value=average_mrr_ltv, dtype=float)
    else:
        ltv_values = np.asarray(sample_weight, dtype=float)

    tp_mask = (y_true_arr == 1) & (y_pred_arr == 1)
    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)
    fn_mask = (y_true_arr == 1) & (y_pred_arr == 0)

    # 1. Net gain from True Positives
    tp_saved_value = np.sum((intervention_success_rate * ltv_values[tp_mask]) - intervention_cost)

    # 2. Wasted spend from False Positives
    fp_wasted_cost = float(np.sum(fp_mask) * intervention_cost)

    # 3. Lost revenue from False Negatives
    fn_lost_value = float(np.sum(ltv_values[fn_mask]))

    total_net_benefit = float(tp_saved_value - fp_wasted_cost - fn_lost_value)
    return total_net_benefit


def create_business_scorer(
    intervention_cost: float = 150.0,
    intervention_success_rate: float = 0.65,
    average_mrr_ltv: float = 10200.0,
) -> Any:
    """Creates a Scikit-Learn compliant Scorer using `make_scorer`.

    Returns:
        _BaseScorer: Scorer object for GridSearchCV / HalvingRandomSearchCV.
    """
    return make_scorer(
        score_func=b2b_churn_net_financial_benefit,
        greater_is_better=True,
        response_method="predict",
        intervention_cost=intervention_cost,
        intervention_success_rate=intervention_success_rate,
        average_mrr_ltv=average_mrr_ltv,
    )


def calculate_financial_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mrr_series: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
    intervention_cost: float = 150.0,
    intervention_success_rate: float = 0.65,
    ltv_mrr_multiplier: float = 12.0,
) -> dict[str, Any]:
    """Evaluates net financial benefits across candidate decision thresholds.

    Args:
        y_true (np.ndarray): Binary ground truth labels.
        y_prob (np.ndarray): Calibrated predicted churn probabilities P(y=1).
        mrr_series (np.ndarray | None): Individual customer MRR values.
        thresholds (np.ndarray | None): Array of candidate thresholds.
        intervention_cost (float): Retention action cost in USD.
        intervention_success_rate (float): Success probability of retention.
        ltv_mrr_multiplier (float): Annual multiplier for MRR to LTV estimation.

    Returns:
        dict[str, Any]: Optimal threshold, maximum benefit, and complete curve data.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if mrr_series is not None:
        ltv_arr = np.asarray(mrr_series, dtype=float) * ltv_mrr_multiplier
    else:
        ltv_arr = np.full_like(y_true_arr, fill_value=850.0 * ltv_mrr_multiplier, dtype=float)

    benefits = []
    tps = []
    fps = []
    fns = []

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        benefit = b2b_churn_net_financial_benefit(
            y_true=y_true_arr,
            y_pred=y_pred,
            sample_weight=ltv_arr,
            intervention_cost=intervention_cost,
            intervention_success_rate=intervention_success_rate,
        )
        benefits.append(benefit)
        tps.append(int(np.sum((y_true_arr == 1) & (y_pred == 1))))
        fps.append(int(np.sum((y_true_arr == 0) & (y_pred == 1))))
        fns.append(int(np.sum((y_true_arr == 1) & (y_pred == 0))))

    benefits_arr = np.array(benefits)
    best_idx = int(np.argmax(benefits_arr))
    best_threshold = float(thresholds[best_idx])
    max_benefit = float(benefits_arr[best_idx])

    return {
        "best_threshold": best_threshold,
        "max_net_benefit_usd": max_benefit,
        "thresholds": thresholds.tolist(),
        "net_benefits": benefits_arr.tolist(),
        "tps_at_best": tps[best_idx],
        "fps_at_best": fps[best_idx],
        "fns_at_best": fns[best_idx],
    }
