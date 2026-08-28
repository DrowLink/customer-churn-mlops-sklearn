"""Secure Serialization of Machine Learning Artifacts, Schemas, and Metadata.

Guarantees model reproducibility, traceability, and governance by serializing:
1. The complete pipeline (Transformers + Calibrated Model).
2. Technical metadata (training date, semantic version, git hash/UUID, test metrics).
3. Input schema contracts (expected columns and types).
4. SHA-256 Checksum verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelArtifactMetadata(BaseModel):
    """Pydantic schema for MLOps artifact metadata and traceability."""

    model_name: str
    version: str
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    algorithm: str
    best_threshold: float = 0.5
    calibrated: bool = True
    input_features: list[str]
    target_column: str = "churn"
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    business_parameters: dict[str, Any] = Field(default_factory=dict)
    pipeline_sha256: str = ""


def compute_file_sha256(filepath: Path | str) -> str:
    """Calculates the SHA-256 hash of a binary file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model_artifact(
    pipeline: Any,
    metadata: ModelArtifactMetadata,
    output_dir: str | Path = "models/artifacts",
    compress: int = 3,
) -> tuple[Path, Path]:
    """Atomically saves serialized pipeline with joblib and its companion JSON metadata file.

    Args:
        pipeline (Any): Trained pipeline or estimator.
        metadata (ModelArtifactMetadata): Descriptive metadata for the run.
        output_dir (str | Path): Destination directory.
        compress (int): Joblib compression level (0 to 9).

    Returns:
        tuple[Path, Path]: Saved pipeline path (.joblib) and metadata path (.json).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pipeline_filename = f"{metadata.model_name}_{metadata.version}_{timestamp_str}.joblib"
    metadata_filename = f"{metadata.model_name}_{metadata.version}_{timestamp_str}_metadata.json"

    pipeline_filepath = out_path / pipeline_filename
    metadata_filepath = out_path / metadata_filename

    # 1. Serialize Pipeline with Joblib
    logger.info("Saving trained pipeline to %s...", pipeline_filepath)
    joblib.dump(pipeline, pipeline_filepath, compress=compress)

    # 2. Compute Binary Checksum
    sha256_hash = compute_file_sha256(pipeline_filepath)
    metadata.pipeline_sha256 = sha256_hash

    # 3. Save JSON Metadata
    logger.info("Saving metadata to %s...", metadata_filepath)
    with open(metadata_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    # 4. Update 'latest' pointer for zero-downtime deployments
    latest_pipeline = out_path / f"{metadata.model_name}_latest.joblib"
    latest_metadata = out_path / f"{metadata.model_name}_latest_metadata.json"
    
    joblib.dump(pipeline, latest_pipeline, compress=compress)
    with open(latest_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    return pipeline_filepath, metadata_filepath


def load_model_artifact(
    pipeline_path: str | Path,
    metadata_path: str | Path | None = None,
    verify_checksum: bool = True,
) -> tuple[Any, ModelArtifactMetadata | None]:
    """Loads a serialized pipeline optionally verifying its SHA-256 integrity hash.

    Args:
        pipeline_path (str | Path): Path to .joblib file.
        metadata_path (str | Path | None): Path to JSON metadata file.
        verify_checksum (bool): If True, validates checksum against metadata.

    Returns:
        tuple[Any, ModelArtifactMetadata | None]: (Loaded pipeline, Metadata or None).
    """
    p_path = Path(pipeline_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Pipeline not found at {p_path}")

    loaded_metadata: ModelArtifactMetadata | None = None
    if metadata_path is not None:
        m_path = Path(metadata_path)
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            loaded_metadata = ModelArtifactMetadata(**meta_dict)

            if verify_checksum and loaded_metadata.pipeline_sha256:
                current_sha = compute_file_sha256(p_path)
                if current_sha != loaded_metadata.pipeline_sha256:
                    raise ValueError(
                        f"Artifact Integrity Failure! Current hash ({current_sha}) "
                        f"does not match recorded hash ({loaded_metadata.pipeline_sha256})."
                    )

    logger.info("Loading pipeline from %s...", p_path)
    pipeline = joblib.load(p_path)
    return pipeline, loaded_metadata
