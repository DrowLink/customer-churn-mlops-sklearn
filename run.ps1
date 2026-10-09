# PowerShell automation script for Customer Churn MLOps developer workflows

param (
    [Parameter(Position=0)]
    [ValidateSet("install", "lint", "format", "test", "cov", "train", "serve", "drift", "clean", "docker-build", "docker-run")]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Customer Churn MLOps CLI Tasks" -ForegroundColor Cyan
    Write-Host "Usage: .\run.ps1 <command>`n"
    Write-Host "Commands:"
    Write-Host "  install       Install project dependencies with uv"
    Write-Host "  lint          Run Ruff linter"
    Write-Host "  format        Format code with Ruff"
    Write-Host "  test          Run pytest test suite"
    Write-Host "  cov           Run pytest with test coverage"
    Write-Host "  train         Train & calibrate model end-to-end"
    Write-Host "  serve         Start FastAPI inference server with auto-reload"
    Write-Host "  drift         Evaluate data and feature drift"
    Write-Host "  clean         Remove Python & pytest caches"
    Write-Host "  docker-build  Build Docker image"
    Write-Host "  docker-run    Run Docker container"
}

switch ($Command) {
    "install" {
        uv pip install -e ".[dev]"
    }
    "lint" {
        uv run --extra dev ruff check .
    }
    "format" {
        uv run --extra dev ruff format .
    }
    "test" {
        uv run --extra dev pytest
    }
    "cov" {
        uv run --extra dev pytest --cov=src --cov-report=term-missing --cov-fail-under=85
    }
    "train" {
        uv run python main.py --config configs/config.yaml
    }
    "serve" {
        uv run python main.py --serve --host 127.0.0.1 --port 8000 --reload
    }
    "drift" {
        uv run python main.py --evaluate-drift
    }
    "clean" {
        Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .ruff_cache, htmlcov -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Remove-Item -Path .coverage -Force -ErrorAction SilentlyContinue
        Write-Host "Caches cleaned." -ForegroundColor Green
    }
    "docker-build" {
        docker build -t customer-churn-mlops:latest .
    }
    "docker-run" {
        docker run -p 8000:8000 -v "${PWD}/models:/app/models" customer-churn-mlops:latest
    }
    default {
        Show-Help
    }
}
