.PHONY: help install lint format test cov train serve drift clean docker-build docker-run

PYTHON ?= python
UV ?= uv

help:
	@echo "Available commands:"
	@echo "  make install      Install project and development dependencies"
	@echo "  make lint         Run Ruff linter"
	@echo "  make format       Format code using Ruff"
	@echo "  make test         Execute pytest test suite"
	@echo "  make cov          Run pytest with coverage report"
	@echo "  make train        Run complete ML training & calibration pipeline"
	@echo "  make serve        Start FastAPI inference microservice"
	@echo "  make drift        Run data & feature drift audit"
	@echo "  make docker-build Build production Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo "  make clean        Remove cache and build artifacts"

install:
	$(UV) pip install -e ".[dev]"

lint:
	$(UV) run --extra dev ruff check .

format:
	$(UV) run --extra dev ruff format .

test:
	$(UV) run --extra dev pytest

cov:
	$(UV) run --extra dev pytest --cov=src --cov-report=term-missing --cov-fail-under=85

train:
	$(UV) run python main.py --config configs/config.yaml

serve:
	$(UV) run python main.py --serve --host 127.0.0.1 --port 8000 --reload

drift:
	$(UV) run python main.py --evaluate-drift

docker-build:
	docker build -t customer-churn-mlops:latest .

docker-run:
	docker run -p 8000:8000 -v ./models:/app/models customer-churn-mlops:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov dist build *.egg-info
