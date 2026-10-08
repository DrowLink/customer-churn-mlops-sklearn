"""FastAPI route handlers for real-time and batch customer churn prediction."""

from __future__ import annotations

import time
from typing import Any
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInferenceRequest,
    FinancialRiskEvaluation,
    HealthResponse,
    SinglePredictionResponse,
)
from src.inference.predict import ChurnInferenceEngine

router = APIRouter()


def get_inference_engine(request: Request) -> ChurnInferenceEngine:
    """Dependency provider for retrieving the loaded ChurnInferenceEngine."""
    engine: ChurnInferenceEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference Engine is not loaded or unavailable.",
        )
    return engine


def calculate_financial_evaluation(
    mrr: float,
    prob: float,
    predicted_churn: bool,
    business_params: dict[str, Any] | None,
) -> FinancialRiskEvaluation:
    """Computes granular financial risk evaluation based on SaaS economics."""
    params = business_params or {}
    ltv_multiplier = float(params.get("ltv_mrr_multiplier", 12.0))
    intervention_cost = float(params.get("intervention_cost_usd", 150.0))
    success_rate = float(params.get("intervention_success_rate", 0.65))

    annual_ltv = mrr * ltv_multiplier
    expected_loss = prob * annual_ltv
    
    if predicted_churn:
        intervention_budget = intervention_cost
        projected_benefit = max(0.0, (success_rate * annual_ltv) - intervention_cost)
    else:
        intervention_budget = 0.0
        projected_benefit = 0.0

    return FinancialRiskEvaluation(
        monthly_recurring_revenue=round(mrr, 2),
        estimated_annual_ltv=round(annual_ltv, 2),
        expected_loss_if_churn=round(expected_loss, 2),
        proactive_intervention_cost=round(intervention_budget, 2),
        projected_net_benefit_usd=round(projected_benefit, 2),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck & Artifact Governance Status",
    description="Returns service health, model version, algorithm, calibration flag, and SHA-256 checksum.",
)
async def health_check(request: Request, engine: ChurnInferenceEngine = Depends(get_inference_engine)) -> HealthResponse:
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())
    meta = engine.metadata
    
    return HealthResponse(
        status="healthy",
        model_name=meta.model_name if meta else "b2b_churn_pipeline",
        model_version=meta.version if meta else "unknown",
        algorithm=meta.algorithm if meta else "unknown",
        calibrated=meta.calibrated if meta else True,
        decision_threshold=round(engine.threshold, 4),
        pipeline_sha256=meta.pipeline_sha256 if meta else "unverified",
        checksum_verified=getattr(request.app.state, "checksum_verified", True),
        uptime_seconds=round(uptime, 2),
    )


@router.post(
    "/v1/predict",
    response_model=SinglePredictionResponse,
    summary="Real-Time Single Customer Churn Prediction",
    description="Evaluates a single customer account payload, returns calibrated churn probability, action guidance, and financial impact.",
)
async def predict_single(
    payload: CustomerInferenceRequest,
    engine: ChurnInferenceEngine = Depends(get_inference_engine),
) -> SinglePredictionResponse:
    # 1. Run inference through engine
    result = engine.predict_single(payload.model_dump())
    
    # 2. Compute financial evaluation
    biz_params = engine.metadata.business_parameters if engine.metadata else None
    predicted_bool = bool(result.churn_prediction == 1)
    fin_eval = calculate_financial_evaluation(
        mrr=payload.monthly_recurring_revenue,
        prob=result.churn_probability,
        predicted_churn=predicted_bool,
        business_params=biz_params,
    )

    return SinglePredictionResponse(
        customer_id=result.customer_id,
        churn_probability=result.churn_probability,
        predicted_churn=predicted_bool,
        decision_threshold_used=result.decision_threshold_used,
        risk_level=result.risk_level,
        recommended_action=result.recommended_action,
        financial_evaluation=fin_eval,
        model_version=engine.metadata.version if engine.metadata else "0.1.0",
    )


@router.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Vectorized Batch Churn Prediction",
    description="Scores a collection of customer accounts in a vectorized fashion and provides aggregated financial risk summaries.",
)
async def predict_batch(
    payload: BatchPredictionRequest,
    engine: ChurnInferenceEngine = Depends(get_inference_engine),
) -> BatchPredictionResponse:
    start_time = time.perf_counter()
    records = [c.model_dump() for c in payload.customers]
    df_input = pd.DataFrame(records)

    # Vectorized inference
    df_scored = engine.batch_predict(df_input)
    biz_params = engine.metadata.business_parameters if engine.metadata else None

    results: list[SinglePredictionResponse] = []
    high_risk_count = 0
    total_mrr_at_risk = 0.0

    for _, row in df_scored.iterrows():
        churn_pred = bool(row["churn_prediction"] == 1)
        prob = float(row["churn_probability"])
        risk = str(row["risk_level"])
        mrr = float(row["monthly_recurring_revenue"])

        if risk in ("HIGH", "CRITICAL"):
            high_risk_count += 1
        if churn_pred:
            total_mrr_at_risk += mrr

        # Recommended action mapping
        if prob < 0.20:
            action = "Maintain standard communication (Routine Nurturing)"
        elif prob < engine.threshold:
            action = "Monitor product adoption and feature usage metrics"
        elif prob < 0.70:
            action = "Trigger proactive Customer Success outreach and alignment call"
        else:
            action = "Immediate executive escalation / Plan renegotiation or custom discount"

        fin_eval = calculate_financial_evaluation(
            mrr=mrr,
            prob=prob,
            predicted_churn=churn_pred,
            business_params=biz_params,
        )

        results.append(
            SinglePredictionResponse(
                customer_id=str(row["customer_id"]),
                churn_probability=prob,
                predicted_churn=churn_pred,
                decision_threshold_used=round(engine.threshold, 4),
                risk_level=risk,  # type: ignore[arg-type]
                recommended_action=action,
                financial_evaluation=fin_eval,
                model_version=engine.metadata.version if engine.metadata else "0.1.0",
            )
        )

    if payload.sort_by_risk:
        results.sort(key=lambda x: x.churn_probability, reverse=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictionResponse(
        total_scored=len(results),
        high_risk_count=high_risk_count,
        total_mrr_at_risk_usd=round(total_mrr_at_risk, 2),
        results=results,
        inference_duration_ms=round(elapsed_ms, 2),
    )
