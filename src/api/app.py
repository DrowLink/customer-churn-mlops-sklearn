"""FastAPI Application Factory and Lifecycle Management."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.inference.artifacts import compute_file_sha256
from src.inference.predict import ChurnInferenceEngine

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_PATH = "models/artifacts/b2b_churn_pipeline_latest.joblib"
DEFAULT_METADATA_PATH = "models/artifacts/b2b_churn_pipeline_latest_metadata.json"


def create_app(
    pipeline_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> FastAPI:
    """Factory function to build and configure the FastAPI application.

    Args:
        pipeline_path (str | Path | None): Optional path to model pipeline joblib file.
        metadata_path (str | Path | None): Optional path to metadata JSON file.

    Returns:
        FastAPI: Fully configured ASGI application.
    """
    pipe_path = Path(
        pipeline_path
        or os.getenv("CHURN_PIPELINE_PATH", DEFAULT_PIPELINE_PATH)
    )
    meta_path = Path(
        metadata_path
        or os.getenv("CHURN_METADATA_PATH", DEFAULT_METADATA_PATH)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager to handle model loading and teardown."""
        app.state.start_time = time.time()
        app.state.engine = None
        app.state.checksum_verified = False

        if pipe_path.exists():
            logger.info("Initializing ChurnInferenceEngine from %s...", pipe_path)
            try:
                engine = ChurnInferenceEngine(
                    pipeline_path=pipe_path,
                    metadata_path=meta_path if meta_path.exists() else None,
                )
                
                # Checksum verification
                if engine.metadata and engine.metadata.pipeline_sha256:
                    actual_hash = compute_file_sha256(pipe_path)
                    if actual_hash == engine.metadata.pipeline_sha256:
                        app.state.checksum_verified = True
                        logger.info("Artifact SHA-256 integrity verified successfully: %s", actual_hash[:16])
                    else:
                        logger.error(
                            "CRITICAL: Artifact checksum mismatch! Expected %s, got %s",
                            engine.metadata.pipeline_sha256,
                            actual_hash,
                        )
                        app.state.checksum_verified = False
                else:
                    app.state.checksum_verified = True

                app.state.engine = engine
                logger.info(
                    "Model %s v%s ready. Decision threshold: %.3f",
                    engine.metadata.model_name if engine.metadata else "b2b_churn",
                    engine.metadata.version if engine.metadata else "latest",
                    engine.threshold,
                )
            except Exception as e:
                logger.exception("Failed to load model pipeline during startup: %s", e)
        else:
            logger.warning(
                "Model pipeline not found at %s. Service running in degraded mode.", pipe_path
            )

        yield

        # Teardown
        logger.info("Shutting down Churn Serving API.")

    app = FastAPI(
        title="B2B Customer Churn MLOps Inference API",
        version="0.1.0",
        description=(
            "Production-grade REST microservice for early B2B SaaS churn prediction, "
            "financial risk evaluation, and proactive Customer Success retention guidance."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routes
    app.include_router(router)

    return app


# Default app instance for ASGI servers like uvicorn src.api.app:app
app = create_app()
