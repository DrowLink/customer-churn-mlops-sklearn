"""Custom Scikit-Learn Transformers for Feature Engineering and Leak-Free Preprocessing.

Implements classes conforming to Scikit-Learn's estimator protocol (`BaseEstimator`, `TransformerMixin`)
to ensure seamless integration within a `Pipeline` or `ColumnTransformer` without Data Leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class B2BRatioFeatureGenerator(BaseEstimator, TransformerMixin):
    """B2B SaaS Business Ratio Feature Generator.

    Calculates key domain metrics from raw product and billing columns:
    1. `seat_utilization_rate` = active_users_last_30d / contracted_seats
    2. `unresolved_tickets_ratio` = unresolved_tickets_count / (support_tickets_count + epsilon)
    3. `avg_logins_per_active_user` = avg_daily_logins / (active_users_last_30d + epsilon)
    4. `mrr_per_contracted_seat` = monthly_recurring_revenue / contracted_seats

    Attributes:
        epsilon (float): Small constant to prevent division by zero.
        drop_intermediates (bool): If True, removes raw base columns after ratio calculation.
    """

    def __init__(self, epsilon: float = 1e-5, drop_intermediates: bool = False) -> None:
        self.epsilon = epsilon
        self.drop_intermediates = drop_intermediates

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> B2BRatioFeatureGenerator:
        """Fits the transformer verifying the presence of all required columns.

        Args:
            X (pd.DataFrame | np.ndarray): Input data.
            y (Any): Ignored (retained for scikit-learn API compatibility).

        Returns:
            B2BRatioFeatureGenerator: Fitted instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"B2BRatioFeatureGenerator requires a pandas DataFrame, received: {type(X)}"
            )

        required_cols = [
            "contracted_seats",
            "active_users_last_30d",
            "support_tickets_count",
            "unresolved_tickets_count",
            "avg_daily_logins",
            "monthly_recurring_revenue",
        ]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in X for ratio calculation: {missing_cols}")

        self.required_cols_ = required_cols
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=str)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes derived ratio features.

        Args:
            X (pd.DataFrame): Input dataframe to transform.

        Returns:
            pd.DataFrame: Dataframe with appended ratio features.
        """
        check_is_fitted(self, ["required_cols_", "feature_names_in_"])
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"B2BRatioFeatureGenerator requires a pandas DataFrame in transform, received: {type(X)}"
            )

        X_out = X.copy()

        # 1. License utilization rate (capped at 1.5 for over-utilization)
        seats = np.maximum(X_out["contracted_seats"].values, 1.0)
        active = np.maximum(X_out["active_users_last_30d"].values, 0.0)
        X_out["seat_utilization_rate"] = np.clip(active / seats, 0.0, 1.5)

        # 2. Unresolved tickets ratio
        tickets = np.maximum(X_out["support_tickets_count"].values, 0.0)
        unresolved = np.maximum(X_out["unresolved_tickets_count"].values, 0.0)
        X_out["unresolved_tickets_ratio"] = np.clip(
            unresolved / (tickets + self.epsilon),
            0.0,
            1.0,
        )

        # 3. Logins per active user
        logins = np.maximum(X_out["avg_daily_logins"].values, 0.0)
        X_out["avg_logins_per_active_user"] = logins / (active + self.epsilon)

        # 4. MRR per contracted seat
        mrr = np.maximum(X_out["monthly_recurring_revenue"].values, 0.0)
        X_out["mrr_per_contracted_seat"] = mrr / seats

        if self.drop_intermediates:
            cols_to_drop = [c for c in self.required_cols_ if c in X_out.columns]
            X_out = X_out.drop(columns=cols_to_drop)

        return X_out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Returns output feature names."""
        check_is_fitted(self, ["feature_names_in_"])
        base_features = list(input_features if input_features is not None else self.feature_names_in_)
        new_features = [
            "seat_utilization_rate",
            "unresolved_tickets_ratio",
            "avg_logins_per_active_user",
            "mrr_per_contracted_seat",
        ]
        if self.drop_intermediates:
            base_features = [f for f in base_features if f not in self.required_cols_]
        return np.array(base_features + new_features, dtype=str)


class RobustOutlierWinsorizer(BaseEstimator, TransformerMixin):
    """Robust quantile-based outlier winsorization transformer.

    Learns clipping thresholds strictly during `.fit()` (on Training set)
    and applies them during `.transform()` without Data Leakage.

    Args:
        lower_quantile (float): Lower quantile bound (default: 0.01).
        upper_quantile (float): Upper quantile bound (default: 0.99).
        columns (Sequence[str] | None): Specific numeric columns to winsorize.
    """

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.columns = columns

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> RobustOutlierWinsorizer:
        """Learns quantile boundaries for each numeric column.

        Args:
            X (pd.DataFrame | np.ndarray): Training data.
            y (Any): Ignored.

        Returns:
            RobustOutlierWinsorizer: Fitted instance.
        """
        if isinstance(X, pd.DataFrame):
            self.columns_to_clip_ = (
                list(self.columns)
                if self.columns is not None
                else list(X.select_dtypes(include=[np.number]).columns)
            )
            self.lower_bounds_ = {
                col: float(X[col].quantile(self.lower_quantile))
                for col in self.columns_to_clip_
                if col in X.columns
            }
            self.upper_bounds_ = {
                col: float(X[col].quantile(self.upper_quantile))
                for col in self.columns_to_clip_
                if col in X.columns
            }
            self.is_dataframe_ = True
            self.feature_names_in_ = np.array(X.columns, dtype=str)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.lower_bounds_arr_ = np.nanquantile(X_arr, self.lower_quantile, axis=0)
            self.upper_bounds_arr_ = np.nanquantile(X_arr, self.upper_quantile, axis=0)
            self.is_dataframe_ = False
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X_arr.shape[1])], dtype=str)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Clips outliers exceeding boundaries learned during fit.

        Args:
            X (pd.DataFrame | np.ndarray): Data to transform.

        Returns:
            pd.DataFrame | np.ndarray: Winsorized output.
        """
        check_is_fitted(self, ["n_features_in_"])

        if self.is_dataframe_ and isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col in self.columns_to_clip_:
                if col in X_out.columns:
                    l_bound = self.lower_bounds_[col]
                    u_bound = self.upper_bounds_[col]
                    X_out[col] = X_out[col].clip(lower=l_bound, upper=u_bound)
            return X_out
        else:
            X_arr = np.asarray(X, dtype=float).copy()
            for col_idx in range(X_arr.shape[1]):
                l_bound = self.lower_bounds_arr_[col_idx]
                u_bound = self.upper_bounds_arr_[col_idx]
                X_arr[:, col_idx] = np.clip(X_arr[:, col_idx], l_bound, u_bound)
            return X_arr

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Returns output feature names."""
        check_is_fitted(self, ["feature_names_in_"])
        if input_features is not None:
            return np.array(input_features, dtype=str)
        return self.feature_names_in_


class SafeLog1pTransformer(BaseEstimator, TransformerMixin):
    """Safely applies log1p transformation to skewed numerical features.

    Guarantees negative values are clipped to zero prior to log transform to prevent NaNs.
    """

    def __init__(self, columns: Sequence[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> SafeLog1pTransformer:
        """Records the input schema."""
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.columns_ = self.columns if self.columns is not None else list(X.columns)
            self.feature_names_in_ = np.array(X.columns, dtype=str)
        else:
            self.columns_ = None
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=str)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Applies log1p with numerical safety bounds."""
        check_is_fitted(self, ["n_features_in_"])
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            target_cols = [c for c in self.columns_ if c in X_out.columns]
            for col in target_cols:
                vals = np.maximum(X_out[col].fillna(0).values, 0.0)
                X_out[col] = np.log1p(vals)
            return X_out
        else:
            X_arr = np.maximum(np.asarray(X, dtype=float), 0.0)
            return np.log1p(X_arr)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Returns output feature names."""
        check_is_fitted(self, ["feature_names_in_"])
        if input_features is not None:
            return np.array(input_features, dtype=str)
        return self.feature_names_in_
