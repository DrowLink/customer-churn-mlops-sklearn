"""Probability Calibration and Dynamic Operational Decision Thresholding.

In class-imbalanced settings with asymmetric financial risk, raw probabilities from tree ensembles
often deviate from true empirical probabilities (especially after using class_weight='balanced').

This module provides:
1. `CalibratedClassifierCV` fitting using 'isotonic' or 'sigmoid' calibration.
2. Brier Score and Expected Calibration Error (ECE) quantitative diagnostics.
3. Dynamic Threshold Optimization maximizing net business benefit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.models.metrics import calculate_financial_curve

logger = logging.getLogger(__name__)


def fit_calibrated_model(
    estimator: BaseEstimator,
    X_train: Any,
    y_train: np.ndarray,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Fits CalibratedClassifierCV on a base estimator or pipeline.

    Args:
        estimator (BaseEstimator): Base scikit-learn estimator or Pipeline.
        X_train (Any): Training feature matrix.
        y_train (np.ndarray): Binary ground truth labels.
        method (str): Calibration method ('isotonic' or 'sigmoid').
        cv (int): Internal cross-validation folds.

    Returns:
        CalibratedClassifierCV: Fitted calibrated classifier.
    """
    logger.info(
        "Calibrating probabilities via CalibratedClassifierCV (method='%s', cv=%d)...", method, cv
    )

    calibrated_clf = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv,
        n_jobs=-1,
    )
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Evaluates the calibration quality of predicted probabilities.

    Calculates:
    - Brier Score Loss (lower is better, 0 = perfect calibration).
    - Expected Calibration Error (ECE): Weighted difference between confidence and accuracy.
    - Calibration curve points (prob_true vs prob_pred).

    Args:
        y_true (np.ndarray): Binary ground truth labels.
        y_prob (np.ndarray): Predicted positive class probabilities P(y=1).
        n_bins (int): Number of bins for calibration curve.
        strategy (str): 'uniform' or 'quantile'.

    Returns:
        dict[str, Any]: Calibration metrics and curve data.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    prob_true, prob_pred = calibration_curve(
        y_true_arr,
        y_prob_arr,
        n_bins=n_bins,
        strategy=strategy,
    )

    # Compute Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob_arr, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    bin_sums = np.bincount(binids, weights=y_prob_arr, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true_arr, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)

    nonzero = bin_total > 0
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_acc[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    bin_conf[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

    ece = float(np.sum(np.abs(bin_acc - bin_conf) * (bin_total / len(y_prob_arr))))

    logger.info("Calibration Diagnostic -> Brier Score: %.4f | ECE: %.4f", brier, ece)

    return {
        "brier_score": brier,
        "ece": ece,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def find_optimal_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mrr_series: np.ndarray | None = None,
    intervention_cost: float = 150.0,
    intervention_success_rate: float = 0.65,
    ltv_mrr_multiplier: float = 12.0,
) -> tuple[float, float, dict[str, Any]]:
    """Finds the optimal decision threshold that maximizes net financial benefit.

    Args:
        y_true (np.ndarray): Binary ground truth labels.
        y_prob (np.ndarray): Calibrated probabilities.
        mrr_series (np.ndarray | None): Individual customer MRR.
        intervention_cost (float): Retention cost per account.
        intervention_success_rate (float): Retention success probability.
        ltv_mrr_multiplier (float): Multiplier to calculate LTV.

    Returns:
        tuple[float, float, dict[str, Any]]: (best_threshold, max_benefit_usd, curve_details).
    """
    curve_data = calculate_financial_curve(
        y_true=y_true,
        y_prob=y_prob,
        mrr_series=mrr_series,
        intervention_cost=intervention_cost,
        intervention_success_rate=intervention_success_rate,
        ltv_mrr_multiplier=ltv_mrr_multiplier,
    )
    best_th = curve_data["best_threshold"]
    max_ben = curve_data["max_net_benefit_usd"]

    logger.info(
        "Optimal Decision Threshold Found: %.3f with Projected Net Benefit of $%.2f USD",
        best_th,
        max_ben,
    )
    return best_th, max_ben, curve_data
