"""Comprehensive unit and integration tests for FastAPI Churn Serving endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def valid_customer_payload() -> dict:
    """Fixture providing a valid enterprise customer inference payload."""
    return {
        "customer_id": "CUST-TEST-ENTERPRISE-01",
        "industry": "FinTech",
        "company_size_bracket": "51-200",
        "plan_tier": "Enterprise",
        "billing_cycle": "Monthly",
        "payment_method": "Bank_Transfer",
        "auto_renew_enabled": False,
        "contract_duration_months": 4,
        "contracted_seats": 80,
        "active_users_last_30d": 15,
        "monthly_recurring_revenue": 5200.0,
        "avg_daily_logins": 10.0,
        "feature_adoption_score": 30.0,
        "storage_usage_pct": 75.0,
        "support_tickets_count": 9,
        "unresolved_tickets_count": 6,
        "nps_score": 3.0,
        "invoice_delay_days": 21.0,
    }


@pytest.fixture(scope="module")
def low_risk_customer_payload() -> dict:
    """Fixture providing a healthy low-churn customer payload."""
    return {
        "customer_id": "CUST-TEST-HEALTHY-02",
        "industry": "Enterprise SaaS",
        "company_size_bracket": "201-1000",
        "plan_tier": "Enterprise",
        "billing_cycle": "Annual",
        "payment_method": "Bank_Transfer",
        "auto_renew_enabled": True,
        "contract_duration_months": 24,
        "contracted_seats": 200,
        "active_users_last_30d": 195,
        "monthly_recurring_revenue": 12000.0,
        "avg_daily_logins": 45.0,
        "feature_adoption_score": 92.0,
        "storage_usage_pct": 80.0,
        "support_tickets_count": 2,
        "unresolved_tickets_count": 0,
        "nps_score": 9.0,
        "invoice_delay_days": 0.0,
    }


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """Initializes TestClient triggering FastAPI lifespan."""
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_check_endpoint(test_client: TestClient) -> None:
    """Verifies /health endpoint returns HTTP 200 and model metadata."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "model_version" in data
    assert "decision_threshold" in data
    assert "pipeline_sha256" in data
    assert data["checksum_verified"] is True
    assert data["uptime_seconds"] >= 0


def test_predict_single_endpoint_happy_path(
    test_client: TestClient,
    valid_customer_payload: dict,
) -> None:
    """Verifies /v1/predict evaluates valid customer payload correctly."""
    response = test_client.post("/v1/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "CUST-TEST-ENTERPRISE-01"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["predicted_churn"], bool)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(data["recommended_action"]) > 0

    fin = data["financial_evaluation"]
    assert fin["monthly_recurring_revenue"] == 5200.0
    assert fin["estimated_annual_ltv"] == 5200.0 * 12.0
    assert fin["expected_loss_if_churn"] >= 0.0
    assert "projected_net_benefit_usd" in fin


def test_predict_single_validation_error_missing_field(test_client: TestClient) -> None:
    """Verifies HTTP 422 when required fields are missing."""
    invalid_payload = {
        "customer_id": "INCOMPLETE-001",
        "industry": "FinTech",
    }
    response = test_client.post("/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_single_validation_error_bounds(
    test_client: TestClient,
    valid_customer_payload: dict,
) -> None:
    """Verifies HTTP 422 when numerical values violate constraints."""
    corrupted = valid_customer_payload.copy()
    corrupted["nps_score"] = 15.0  # NPS must be <= 10.0
    response = test_client.post("/v1/predict", json=corrupted)
    assert response.status_code == 422

    corrupted_mrr = valid_customer_payload.copy()
    corrupted_mrr["monthly_recurring_revenue"] = -50.0  # MRR must be >= 0.0
    response = test_client.post("/v1/predict", json=corrupted_mrr)
    assert response.status_code == 422


def test_predict_batch_endpoint(
    test_client: TestClient,
    valid_customer_payload: dict,
    low_risk_customer_payload: dict,
) -> None:
    """Verifies /v1/predict/batch vector scoring and risk ranking."""
    batch_payload = {
        "customers": [valid_customer_payload, low_risk_customer_payload],
        "sort_by_risk": True,
    }
    response = test_client.post("/v1/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_scored"] == 2
    assert "high_risk_count" in data
    assert "total_mrr_at_risk_usd" in data
    assert len(data["results"]) == 2
    assert data["inference_duration_ms"] >= 0

    # Verification of sorting descending by probability
    probs = [r["churn_probability"] for r in data["results"]]
    assert probs == sorted(probs, reverse=True)


def test_service_unavailable_without_model() -> None:
    """Verifies HTTP 503 is returned if engine is not loaded."""
    # Point to nonexistent model path
    app = create_app(
        pipeline_path="models/artifacts/non_existent.joblib",
        metadata_path="models/artifacts/non_existent.json",
    )
    with TestClient(app) as client:
        # Health still works and reports degraded
        health_resp = client.get("/health")
        assert health_resp.status_code == 503

        # Predict returns 503 Service Unavailable
        pred_resp = client.post(
            "/v1/predict",
            json={
                "customer_id": "CUST-01",
                "industry": "FinTech",
                "company_size_bracket": "1-10",
                "plan_tier": "Starter",
                "billing_cycle": "Monthly",
                "payment_method": "ACH",
                "auto_renew_enabled": True,
                "contract_duration_months": 1,
                "contracted_seats": 1,
                "active_users_last_30d": 1,
                "monthly_recurring_revenue": 100.0,
                "avg_daily_logins": 1.0,
                "feature_adoption_score": 10.0,
                "support_tickets_count": 0,
                "unresolved_tickets_count": 0,
                "invoice_delay_days": 0.0,
            },
        )
        assert pred_resp.status_code == 503
