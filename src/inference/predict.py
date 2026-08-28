"""Inference Module for Real-Time (Online) and Batch Predictions.

Implements input data validation using Pydantic, dynamic threshold decisioning,
and structured production output schemas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.inference.artifacts import ModelArtifactMetadata, load_model_artifact

logger = logging.getLogger(__name__)


class CustomerChurnInputSchema(BaseModel):
    """Input data contract for single-customer real-time inference."""

    customer_id: str = "CUST-ONLINE-001"
    industry: Literal["FinTech", "HealthTech", "E-commerce", "EdTech", "Enterprise SaaS", "Logistics"]
    company_size_bracket: Literal["1-10", "11-50", "51-200", "201-1000", "1000+"]
    plan_tier: Literal["Starter", "Professional", "Enterprise"]
    billing_cycle: Literal["Monthly", "Annual", "Multi-Year"]
    payment_method: Literal["Credit_Card", "Bank_Transfer", "ACH"]
    auto_renew_enabled: bool
    contract_duration_months: int = Field(ge=1, le=120)
    contracted_seats: int = Field(ge=1)
    active_users_last_30d: int = Field(ge=0)
    monthly_recurring_revenue: float = Field(ge=0.0)
    avg_daily_logins: float = Field(ge=0.0)
    feature_adoption_score: float = Field(ge=0.0, le=100.0)
    storage_usage_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    support_tickets_count: int = Field(ge=0)
    unresolved_tickets_count: int = Field(ge=0)
    nps_score: float | None = Field(default=None, ge=0.0, le=10.0)
    invoice_delay_days: float = Field(ge=0.0)


class ChurnPredictionOutput(BaseModel):
    """Structured response schema for churn predictions."""

    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str
    decision_threshold_used: float


class ChurnInferenceEngine:
    """Production Inference Engine for B2B Churn Models."""

    def __init__(
        self,
        pipeline_path: str | Path,
        metadata_path: str | Path | None = None,
        override_threshold: float | None = None,
    ) -> None:
        self.pipeline, self.metadata = load_model_artifact(
            pipeline_path=pipeline_path,
            metadata_path=metadata_path,
        )
        if override_threshold is not None:
            self.threshold = override_threshold
        elif self.metadata is not None:
            self.threshold = self.metadata.best_threshold
        else:
            self.threshold = 0.5
            
        logger.info("Inference Engine initialized. Active decision threshold: %.3f", self.threshold)

    def predict_single(self, customer_input: CustomerChurnInputSchema | dict[str, Any]) -> ChurnPredictionOutput:
        """Executes real-time inference for a single customer payload.

        Args:
            customer_input (CustomerChurnInputSchema | dict[str, Any]): Customer metrics.

        Returns:
            ChurnPredictionOutput: Structured prediction with risk tier and suggested retention action.
        """
        if isinstance(customer_input, dict):
            validated_input = CustomerChurnInputSchema(**customer_input)
        else:
            validated_input = customer_input

        df_input = pd.DataFrame([validated_input.model_dump()])
        prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        prediction = int(prob >= self.threshold)

        # Risk categorization and business recommendation
        if prob < 0.20:
            risk = "LOW"
            action = "Maintain standard communication (Routine Nurturing)"
        elif prob < self.threshold:
            risk = "MEDIUM"
            action = "Monitor product adoption and feature usage metrics"
        elif prob < 0.70:
            risk = "HIGH"
            action = "Trigger proactive Customer Success outreach and alignment call"
        else:
            risk = "CRITICAL"
            action = "Immediate executive escalation / Plan renegotiation or custom discount"

        return ChurnPredictionOutput(
            customer_id=validated_input.customer_id,
            churn_probability=round(prob, 4),
            churn_prediction=prediction,
            risk_level=risk,
            recommended_action=action,
            decision_threshold_used=round(self.threshold, 4),
        )

    def batch_predict(
        self,
        df_batch: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Executes batch inference over large customer datasets.

        Args:
            df_batch (pd.DataFrame): Dataframe of customers to evaluate.
            output_path (str | Path | None): Optional destination path (CSV or Parquet).

        Returns:
            pd.DataFrame: Original dataframe enriched with scores, predictions, and risk tiers.
        """
        logger.info("Running batch inference on %d records...", len(df_batch))
        df_result = df_batch.copy()

        # Calibrated probability inference
        probs = self.pipeline.predict_proba(df_batch)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        df_result["churn_probability"] = np.round(probs, 4)
        df_result["churn_prediction"] = preds
        df_result["decision_threshold"] = round(self.threshold, 4)

        # Vectorized Risk Tiering
        conditions = [
            df_result["churn_probability"] < 0.20,
            df_result["churn_probability"] < self.threshold,
            df_result["churn_probability"] < 0.70,
        ]
        choices = ["LOW", "MEDIUM", "HIGH"]
        df_result["risk_level"] = np.select(conditions, choices, default="CRITICAL")

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix == ".parquet":
                try:
                    df_result.to_parquet(out_p, index=False)
                except ImportError:
                    csv_fallback = out_p.with_suffix(".csv")
                    df_result.to_csv(csv_fallback, index=False)
                    logger.warning("pyarrow not installed. Exported as CSV to %s", csv_fallback)
            else:
                df_result.to_csv(out_p, index=False)
            logger.info("Batch prediction results saved to %s", out_p)

        return df_result
