"""Tests para métricas de negocio, optimización de thresholds y calibración."""

import numpy as np

from src.evaluation.calibration import evaluate_calibration, find_optimal_decision_threshold
from src.models.metrics import b2b_churn_net_financial_benefit


def test_b2b_churn_net_financial_benefit_calculation():
    # Caso 1: Detección perfecta (1 TP, 0 FP, 0 FN)
    # TP: gana (0.65 * 10,000) - 150 = 6,500 - 150 = 6,350
    y_true = np.array([1])
    y_pred = np.array([1])
    sample_weight = np.array([10000.0])
    benefit = b2b_churn_net_financial_benefit(
        y_true, y_pred, sample_weight, intervention_cost=150.0, intervention_success_rate=0.65
    )
    assert benefit == 6350.0

    # Caso 2: Falso Positivo (1 FP)
    # FP: costo desperdiciado -150
    y_true_fp = np.array([0])
    y_pred_fp = np.array([1])
    benefit_fp = b2b_churn_net_financial_benefit(
        y_true_fp, y_pred_fp, sample_weight, intervention_cost=150.0
    )
    assert benefit_fp == -150.0


def test_find_optimal_decision_threshold():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.85, 0.7, 0.1, 0.2, 0.3, 0.65, 0.15, 0.8, 0.25])
    mrr = np.array([500.0] * 10)

    best_th, max_benefit, curve_data = find_optimal_decision_threshold(
        y_true=y_true,
        y_prob=y_prob,
        mrr_series=mrr,
        intervention_cost=150.0,
        intervention_success_rate=0.65,
    )

    assert 0.05 <= best_th <= 0.95
    assert isinstance(max_benefit, float)
    assert len(curve_data["thresholds"]) == len(curve_data["net_benefits"])


def test_evaluate_calibration():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.75, 0.25])

    res = evaluate_calibration(y_true, y_prob, n_bins=5)
    assert "brier_score" in res
    assert "ece" in res
    assert 0.0 <= res["brier_score"] <= 1.0
    assert 0.0 <= res["ece"] <= 1.0
