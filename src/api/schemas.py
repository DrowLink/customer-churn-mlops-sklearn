"""Pydantic schemas and data contracts for the FastAPI serving layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class CustomerInferenceRequest(BaseModel):
    """Input payload for scoring a single B2B SaaS customer account."""

    customer_id: str = Field(
        default="CUST-ENT-001",
        description="Unique identifier for the customer account in CRM/Billing.",
    )
    industry: Literal[
        "FinTech", "HealthTech", "E-commerce", "EdTech", "Enterprise SaaS", "Logistics"
    ] = Field(..., description="Vertical industry of the customer organization.")
    company_size_bracket: Literal["1-10", "11-50", "51-200", "201-1000", "1000+"] = Field(
        ..., description="Employee count bracket."
    )
    plan_tier: Literal["Starter", "Professional", "Enterprise"] = Field(
        ..., description="Current contracted subscription tier."
    )
    billing_cycle: Literal["Monthly", "Annual", "Multi-Year"] = Field(
        ..., description="Billing recurrence cadence."
    )
    payment_method: Literal["Credit_Card", "Bank_Transfer", "ACH"] = Field(
        ..., description="Primary payment rails utilized."
    )
    auto_renew_enabled: bool = Field(
        ..., description="Whether contract renewal is configured to auto-charge."
    )
    contract_duration_months: int = Field(
        ..., ge=1, le=120, description="Months elapsed since initial account activation."
    )
    contracted_seats: int = Field(..., ge=1, description="Total licensed user seats purchased.")
    active_users_last_30d: int = Field(
        ..., ge=0, description="Unique licensed users with at least 1 login in last 30 days."
    )
    monthly_recurring_revenue: float = Field(
        ..., ge=0.0, description="Normalized Monthly Recurring Revenue (MRR) in USD."
    )
    avg_daily_logins: float = Field(
        ..., ge=0.0, description="Average cumulative session logins per day across all users."
    )
    feature_adoption_score: float = Field(
        ..., ge=0.0, le=100.0, description="Standardized product feature breadth score (0-100)."
    )
    storage_usage_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percentage of contracted cloud storage consumed.",
    )
    support_tickets_count: int = Field(
        ..., ge=0, description="Total support tickets filed over the previous 90-day window."
    )
    unresolved_tickets_count: int = Field(
        ..., ge=0, description="Currently open or unresolved support tickets."
    )
    nps_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Most recent Net Promoter Score submission (0-10).",
    )
    invoice_delay_days: float = Field(
        ..., ge=0.0, description="Average calendar days payment was delayed beyond net terms."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-ENT-FIN-9921",
                "industry": "FinTech",
                "company_size_bracket": "51-200",
                "plan_tier": "Enterprise",
                "billing_cycle": "Monthly",
                "payment_method": "Bank_Transfer",
                "auto_renew_enabled": False,
                "contract_duration_months": 5,
                "contracted_seats": 100,
                "active_users_last_30d": 18,
                "monthly_recurring_revenue": 6250.0,
                "avg_daily_logins": 7.0,
                "feature_adoption_score": 24.5,
                "storage_usage_pct": 72.0,
                "support_tickets_count": 11,
                "unresolved_tickets_count": 7,
                "nps_score": 2.5,
                "invoice_delay_days": 19.0,
            }
        }
    }


class FinancialRiskEvaluation(BaseModel):
    """Financial impact breakdown based on SaaS LTV and Customer Success economics."""

    monthly_recurring_revenue: float = Field(description="Customer MRR in USD.")
    estimated_annual_ltv: float = Field(
        description="Estimated remaining Lifetime Value based on MRR multiplier."
    )
    expected_loss_if_churn: float = Field(
        description="Net expected financial loss if customer terminates contract."
    )
    proactive_intervention_cost: float = Field(
        description="Budget allocation for Customer Success proactive engagement."
    )
    projected_net_benefit_usd: float = Field(
        description="Estimated net financial benefit of successful retention."
    )


class SinglePredictionResponse(BaseModel):
    """Enriched response for a single customer churn prediction."""

    customer_id: str
    churn_probability: float = Field(description="Calibrated probability of churn (0.0 to 1.0).")
    predicted_churn: bool = Field(description="Binary classification decision based on threshold.")
    decision_threshold_used: float = Field(description="Optimal decision threshold applied.")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Assigned risk tier based on probability and financial stakes."
    )
    recommended_action: str = Field(
        description="Actionable guidance for the Customer Success team."
    )
    financial_evaluation: FinancialRiskEvaluation
    model_version: str
    inference_timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class BatchPredictionRequest(BaseModel):
    """Request payload for multi-customer batch scoring."""

    customers: list[CustomerInferenceRequest] = Field(
        ..., min_length=1, description="List of customer records to evaluate."
    )
    sort_by_risk: bool = Field(
        default=True, description="Whether to sort output descending by churn probability."
    )


class BatchPredictionResponse(BaseModel):
    """Response payload for multi-customer batch scoring."""

    total_scored: int = Field(description="Total number of evaluated accounts.")
    high_risk_count: int = Field(description="Count of accounts flagged as HIGH or CRITICAL risk.")
    total_mrr_at_risk_usd: float = Field(description="Total MRR of accounts predicted to churn.")
    results: list[SinglePredictionResponse] = Field(
        description="Individual account prediction details."
    )
    inference_duration_ms: float = Field(
        description="Elapsed server execution time in milliseconds."
    )


class HealthResponse(BaseModel):
    """Health check and model governance status response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_name: str
    model_version: str
    algorithm: str
    calibrated: bool
    decision_threshold: float
    pipeline_sha256: str
    checksum_verified: bool
    uptime_seconds: float
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
