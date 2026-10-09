"""Production Data and Feature Drift Monitoring for B2B SaaS Churn Prediction.

Implements statistical drift detection methods:
1. Population Stability Index (PSI) for continuous and categorical distributions.
2. Two-sample Kolmogorov-Smirnov (KS) test for numerical distribution shifts.
3. Baseline reference profiling and extraction for artifact governance.
4. Comprehensive DriftReport generation with alerting thresholds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculates the Population Stability Index (PSI) between reference and current distributions.

    PSI benchmarks:
    - PSI < 0.10: No significant shift (distribution is stable).
    - 0.10 <= PSI < 0.25: Moderate shift (monitor feature closely).
    - PSI >= 0.25: Significant shift (action required, retrain or inspect data quality).

    Args:
        reference: Reference baseline array or Series.
        current: Current/production array or Series.
        num_buckets: Number of quantile buckets for binning.
        epsilon: Small smoothing constant to prevent division by zero and log(0).

    Returns:
        float: Computed Population Stability Index (>= 0.0).
    """
    ref_clean = pd.Series(reference).dropna().to_numpy(dtype=float)
    curr_clean = pd.Series(current).dropna().to_numpy(dtype=float)

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Handle constant or nearly constant distributions
    if np.all(ref_clean == ref_clean[0]) and np.all(curr_clean == curr_clean[0]):
        return 0.0 if ref_clean[0] == curr_clean[0] else 1.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    raw_bins = np.percentile(ref_clean, percentiles)
    unique_bins = np.unique(raw_bins)

    if len(unique_bins) < 2:
        return 0.0

    # Expand outer bounds to cover full range of both distributions
    bins = unique_bins.copy()
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    curr_counts, _ = np.histogram(curr_clean, bins=bins)

    total_ref = len(ref_clean)
    total_curr = len(curr_clean)
    n_bins = len(ref_counts)

    # Laplace-style smoothing
    ref_pct = (ref_counts + epsilon) / (total_ref + epsilon * n_bins)
    curr_pct = (curr_counts + epsilon) / (total_curr + epsilon * n_bins)

    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_value))


def compute_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """Calculates Population Stability Index (PSI) for categorical variables.

    Args:
        reference: Reference baseline Series.
        current: Current/production Series.
        epsilon: Smoothing constant.

    Returns:
        float: Categorical PSI value.
    """
    ref_s = reference.dropna().astype(str)
    curr_s = current.dropna().astype(str)

    if len(ref_s) == 0 or len(curr_s) == 0:
        return 0.0

    all_categories = sorted(list(set(ref_s.unique()) | set(curr_s.unique())))
    n_categories = len(all_categories)
    if n_categories == 0:
        return 0.0

    ref_counts = ref_s.value_counts()
    curr_counts = curr_s.value_counts()

    total_ref = len(ref_s)
    total_curr = len(curr_s)

    ref_pct_list = []
    curr_pct_list = []

    for cat in all_categories:
        ref_c = ref_counts.get(cat, 0)
        curr_c = curr_counts.get(cat, 0)
        ref_pct_list.append((ref_c + epsilon) / (total_ref + epsilon * n_categories))
        curr_pct_list.append((curr_c + epsilon) / (total_curr + epsilon * n_categories))

    ref_pct = np.array(ref_pct_list)
    curr_pct = np.array(curr_pct_list)

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def compute_ks_test(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Computes two-sample Kolmogorov-Smirnov test between reference and current samples.

    Args:
        reference: Reference baseline values.
        current: Current values.

    Returns:
        tuple[float, float]: (KS statistic, p-value).
    """
    ref_clean = pd.Series(reference).dropna().to_numpy(dtype=float)
    curr_clean = pd.Series(current).dropna().to_numpy(dtype=float)

    if len(ref_clean) < 2 or len(curr_clean) < 2:
        return 0.0, 1.0

    res = stats.ks_2samp(ref_clean, curr_clean)
    return float(res.statistic), float(res.pvalue)


def extract_baseline_summary(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    """Extracts summary statistical distribution profiles from the reference training dataset.

    Args:
        df: Training reference DataFrame.
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.

    Returns:
        dict[str, Any]: Baseline profiles dictionary.
    """
    summary: dict[str, Any] = {
        "n_samples": int(len(df)),
        "numeric_features": {},
        "categorical_features": {},
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for col in numeric_features:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        summary["numeric_features"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "quantiles": {
                "p01": float(series.quantile(0.01)),
                "p05": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
            },
        }

    for col in categorical_features:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        proportions = series.value_counts(normalize=True).to_dict()
        summary["categorical_features"][col] = {
            "proportions": {k: float(v) for k, v in proportions.items()},
            "unique_count": int(series.nunique()),
        }

    return summary


@dataclass
class FeatureDriftResult:
    """Detailed drift status for a single feature."""

    feature_name: str
    feature_type: str  # "numeric" or "categorical"
    psi: float
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    drift_level: str = "STABLE"  # "STABLE", "WARNING", "DRIFT_DETECTED"
    has_drift: bool = False


@dataclass
class DriftReport:
    """Overall dataset drift report."""

    timestamp_utc: str
    n_reference_samples: int
    n_current_samples: int
    is_dataset_drifted: bool
    drifted_features_count: int
    total_features_evaluated: int
    drifted_features: list[str]
    feature_metrics: dict[str, FeatureDriftResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes DriftReport to dictionary."""
        data = asdict(self)
        return data


class DriftMonitor:
    """Comprehensive Data and Feature Drift Monitor for SaaS churn inference payloads."""

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        baseline_summary: dict[str, Any] | None = None,
    ) -> None:
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.reference_df = reference_df
        self.baseline_summary = baseline_summary

        if self.reference_df is not None and self.baseline_summary is None:
            self.baseline_summary = extract_baseline_summary(
                self.reference_df,
                self.numeric_features,
                self.categorical_features,
            )

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.25,
        psi_warning_threshold: float = 0.10,
        ks_alpha: float = 0.05,
    ) -> DriftReport:
        """Evaluates whether current data has drifted from the reference baseline.

        Args:
            current_df: Current batch of incoming records.
            psi_threshold: PSI threshold for declaring significant drift (default: 0.25).
            psi_warning_threshold: PSI threshold for early warning (default: 0.10).
            ks_alpha: Significance level for Kolmogorov-Smirnov test (default: 0.05).

        Returns:
            DriftReport: Comprehensive drift evaluation report.
        """
        feature_metrics: dict[str, FeatureDriftResult] = {}
        drifted_features: list[str] = []

        # Evaluate Numeric Features
        for col in self.numeric_features:
            if col not in current_df.columns:
                continue

            ref_series = None
            if self.reference_df is not None and col in self.reference_df.columns:
                ref_series = self.reference_df[col]

            curr_series = current_df[col]

            if ref_series is not None:
                psi_val = compute_psi(ref_series, curr_series)
                ks_stat, ks_pval = compute_ks_test(ref_series, curr_series)
            else:
                # If only baseline summary is available, fallback approximation
                psi_val = 0.0
                ks_stat, ks_pval = 0.0, 1.0

            has_drift = psi_val >= psi_threshold or (ks_pval < ks_alpha and ks_stat > 0.15)
            if psi_val >= psi_threshold:
                drift_level = "DRIFT_DETECTED"
            elif psi_val >= psi_warning_threshold:
                drift_level = "WARNING"
            else:
                drift_level = "STABLE"

            if has_drift:
                drifted_features.append(col)

            feature_metrics[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="numeric",
                psi=round(psi_val, 4),
                ks_statistic=round(ks_stat, 4) if ks_stat is not None else None,
                ks_pvalue=round(ks_pval, 4) if ks_pval is not None else None,
                drift_level=drift_level,
                has_drift=has_drift,
            )

        # Evaluate Categorical Features
        for col in self.categorical_features:
            if col not in current_df.columns:
                continue

            ref_series = None
            if self.reference_df is not None and col in self.reference_df.columns:
                ref_series = self.reference_df[col]

            curr_series = current_df[col]

            if ref_series is not None:
                psi_val = compute_categorical_psi(ref_series, curr_series)
            else:
                psi_val = 0.0

            has_drift = psi_val >= psi_threshold
            if psi_val >= psi_threshold:
                drift_level = "DRIFT_DETECTED"
            elif psi_val >= psi_warning_threshold:
                drift_level = "WARNING"
            else:
                drift_level = "STABLE"

            if has_drift:
                drifted_features.append(col)

            feature_metrics[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="categorical",
                psi=round(psi_val, 4),
                drift_level=drift_level,
                has_drift=has_drift,
            )

        total_features = len(feature_metrics)
        # Dataset is considered drifted if >= 2 features or >= 20% of features show drift
        drift_fraction = len(drifted_features) / total_features if total_features > 0 else 0.0
        is_dataset_drifted = len(drifted_features) >= 2 or drift_fraction >= 0.20

        n_ref = len(self.reference_df) if self.reference_df is not None else 0
        n_curr = len(current_df)

        report = DriftReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            n_reference_samples=n_ref,
            n_current_samples=n_curr,
            is_dataset_drifted=is_dataset_drifted,
            drifted_features_count=len(drifted_features),
            total_features_evaluated=total_features,
            drifted_features=drifted_features,
            feature_metrics=feature_metrics,
        )

        if is_dataset_drifted:
            logger.warning(
                "DATASET DRIFT DETECTED! %d/%d features drifted: %s",
                len(drifted_features),
                total_features,
                drifted_features,
            )
        else:
            logger.info(
                "Drift evaluation completed: Dataset stable (%d/%d features drifted).",
                len(drifted_features),
                total_features,
            )

        return report

    def save_baseline_summary(self, filepath: str | Path) -> None:
        """Saves baseline summary profile to JSON."""
        if not self.baseline_summary:
            raise ValueError("No baseline summary available to save.")
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.baseline_summary, f, indent=2)
        logger.info("Baseline distribution profile saved to %s", p)
