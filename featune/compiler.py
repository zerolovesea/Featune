# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Fold-local feature compilation and sklearn preprocessing.

Implements training-only stateful transformations and sklearn preprocessing.
Inference reuses fitted statistics and never learns from validation/test rows.

Created:
    2026-09-21
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .ir import FeatureSet
from .schema import DatasetSchema, FeatureSpec, validate_features


def category_key(series):
    # Prefix values so a literal missing marker cannot collide with missing data.
    """Encode missing and present categories without missing-marker collisions.

    Args:
        series (pandas.Series): Categorical values with scalar missingness.

    Returns:
        pandas.Series: String keys with N: for missing values and V: for present values.

    Notes:
        Present values include their Python type so mixed-type categories remain distinct.
    """
    return series.map(
        lambda value: (
            "N:" if pd.isna(value) else f"V:{type(value).__module__}.{type(value).__qualname__}:{value}"
        )
    )


class FeatureCompiler(TransformerMixin, BaseEstimator):
    """Fit and apply closed-DSL features as a sklearn DataFrame transformer.

    Attributes:
        schema (DatasetSchema): Raw-field contract supplied at construction.
        features (FeatureSet or sequence or None): Unfitted derived-feature definitions.
        preserve_missing_categories (bool): Keep categorical nulls for native missing-value models.
        feature_set_ (FeatureSet): Normalized graph created by fit.
        expression_hashes_ (dict[str, str]): Recursive expression identities used for cross
            canonicalization.
        types_ (dict[str, str]): Output dtypes in compiled column order.
        statistics_ (dict): Training-only frequency/group mappings and fallback values.
        feature_names_in_ (numpy.ndarray): Usable raw column names.
        n_features_in_ (int): Count of usable raw columns.

    Notes:
        Stateful operations learn from fit rows only. transform retains row indices and
        emits original usable columns followed by derived columns.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        features: list[FeatureSpec] | None = None,
        preserve_missing_categories=False,
    ):
        """Store sklearn-cloneable schema and feature parameters.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            features (FeatureSet, sequence[FeatureSpec], or None): Derived expressions; None means raw
                fields only.
            preserve_missing_categories (bool): Preserve categorical NaN at output; False keeps
                the historical explicit missing-category key. Learned DSL grouping is unchanged.
        """
        self.schema = schema
        self.features = features
        self.preserve_missing_categories = preserve_missing_categories

    def fit(self, X, y=None):
        """Validate inputs and learn stateful statistics from these training rows.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            y (array-like or None): Ignored; feature statistics are target-free.

        Returns:
            FeatureCompiler: This fitted transformer.

        Raises:
            ValueError: Frame or feature contracts are invalid.

        Notes:
            Replaces previously fitted statistics. Call separately inside each CV training fold.
        """
        self.schema.validate_frame(X)
        self.feature_set_ = FeatureSet.from_features(self.schema, self.features or [])
        self.expression_hashes_ = self.feature_set_.expression_hashes()
        self.types_ = validate_features(self.schema, self.feature_set_.to_specs())
        self.statistics_ = {}
        self.feature_names_in_ = np.asarray(list(self.schema.usable), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self._compile(X, fitting=True)
        return self

    def transform(self, X):
        """Apply fitted transformations without updating training statistics.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.

        Returns:
            pandas.DataFrame: Float numeric and encoded categorical columns aligned with X.

        Raises:
            NotFittedError: fit has not initialized the compiler.
            ValueError: X violates the fitted schema.
        """
        check_is_fitted(self, "statistics_")
        self.schema.validate_frame(X)
        return self._compile(X, fitting=False)

    def get_feature_names_out(self, input_features=None):
        """Expose compiled column names in execution order.

        Args:
            input_features (array-like or None): Ignored; names are determined by the schema and feature
                graph.

        Returns:
            numpy.ndarray: One-dimensional object array of output names.

        Raises:
            NotFittedError: The compiler has not been fitted.
        """
        check_is_fitted(self, "types_")
        return np.asarray(list(self.types_), dtype=object)

    def _compile(self, X, fitting):
        # Construct numeric columns together to avoid fragmentation on wide tables.
        """Execute validated operators and normalize model-facing values.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            fitting (bool): Learn training-only state when True; otherwise reuse previously fitted state.

        Returns:
            pandas.DataFrame: New compiled frame preserving the input index.

        Notes:
            fitting=True updates statistics_; inference only reads it. Integer inputs are
            promoted before arithmetic. Unknown groups use training fallbacks, near-zero
            division becomes missing, and non-finite numeric outputs become NaN.
        """
        frame = pd.DataFrame(
            {
                name: X[name].astype(float) if field.dtype == "numeric" else X[name]
                for name, field in self.schema.usable.items()
            },
            index=X.index,
        )
        for name, field in self.schema.usable.items():
            if field.dtype == "datetime":
                frame[name] = pd.to_datetime(frame[name], utc=True)
            elif field.dtype == "categorical":
                frame[name] = frame[name].astype(object).where(frame[name].notna(), np.nan)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for feature in self.feature_set_.features:
                inputs = feature.inputs
                if feature.op == "cross":
                    # Cross is a commutative DSL expression; its encoding must be canonical too.
                    inputs = sorted(
                        inputs,
                        key=lambda name: self.expression_hashes_.get(name, feature.lineage_parents[name]),
                    )
                values = [frame[name] for name in inputs]
                a = values[0]
                op = feature.op
                if op in {"add", "subtract", "multiply", "divide"}:
                    b = values[1]
                    if op == "add":
                        value = a + b
                    elif op == "subtract":
                        value = a - b
                    elif op == "multiply":
                        value = a * b
                    else:
                        value = a / b.where(b.abs() > 1e-12)
                elif op == "log1p_abs":
                    value = np.log1p(a.abs())
                elif op == "sqrt_abs":
                    value = np.sqrt(a.abs())
                elif op == "square":
                    value = a**2
                elif op == "abs":
                    value = a.abs()
                elif op == "clip":
                    value = a.clip(**feature.params)
                elif op == "is_missing":
                    value = a.isna().astype(float)
                elif op == "cross":
                    # Length prefix makes the categorical pair encoding injective.
                    left, right = category_key(a), category_key(values[1])
                    value = left.str.len().astype(str) + ":" + left + right
                elif op in {"frequency", "group_mean", "group_std"}:
                    key = category_key(a)
                    if fitting:
                        if op == "frequency":
                            mapping, fallback = key.value_counts(normalize=True), 0.0
                        else:
                            numeric = values[1].replace([np.inf, -np.inf], np.nan)
                            grouped = numeric.groupby(key)
                            mapping = grouped.mean() if op == "group_mean" else grouped.std(ddof=0)
                            fallback = numeric.mean() if op == "group_mean" else numeric.std(ddof=0)
                        self.statistics_[feature.name] = (mapping, fallback)
                    mapping, fallback = self.statistics_[feature.name]
                    value = key.map(mapping).fillna(fallback)
                elif op == "days_between":
                    value = (a - values[1]).dt.total_seconds() / 86400
                else:
                    value = getattr(a.dt, op).astype(float)
                frame[feature.name] = value
        for name, dtype in self.types_.items():
            if dtype == "datetime":
                dates = frame[name]
                frame[name] = dates.map(lambda d: d.timestamp() / 86400 if pd.notna(d) else np.nan)
            elif dtype == "categorical":
                values = frame[name]
                frame[name] = category_key(values)
                if self.preserve_missing_categories:
                    frame[name] = frame[name].where(values.notna(), np.nan)
            else:
                frame[name] = frame[name].astype(float).replace([np.inf, -np.inf], np.nan)
        return frame


def build_pipeline(schema, features, estimator):
    """Assemble fold-local feature compilation, preprocessing and prediction.

    Args:
        schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
        features (FeatureSet or sequence[FeatureSpec]): Controlled derived features, excluding raw
            columns.
        estimator (sklearn estimator): Cloneable prediction model; set its random seed for
            reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted pipeline owning the supplied estimator.

    Notes:
        DataFrame-aware estimators receive compiled columns directly. Other estimators
        receive median-imputed/scaled numeric columns and bounded dense one-hot categories.
        The caller must clone estimator when isolation from an existing model is required.
    """
    features = FeatureSet.from_features(schema, features)
    types = validate_features(schema, features.to_specs())
    numeric = [name for name, dtype in types.items() if dtype != "categorical"]
    categorical = [name for name, dtype in types.items() if dtype == "categorical"]
    if getattr(estimator, "accepts_dataframe", False):
        compiler = FeatureCompiler(
            schema,
            features,
            preserve_missing_categories=getattr(estimator, "preserve_missing_categories", False),
        )
        return Pipeline([("features", compiler), ("model", estimator)])
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", max_categories=128, sparse_output=False),
                categorical,
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [("features", FeatureCompiler(schema, features)), ("preprocess", preprocessor), ("model", estimator)]
    )
