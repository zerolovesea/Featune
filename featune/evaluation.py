# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Independent, paired-fold evaluation and numerical evidence.

Owns paired folds, numerical evaluation and permutation evidence. Samplers do
not select folds or access the external holdout through this module.

Created:
    2026-09-21
"""

import logging
import time
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .compiler import build_pipeline
from .ir import COMPILER_VERSION, FeatureSet, content_hash
from .tabpfn import _TabPFNBase, default_estimator

METRICS = {
    "auc": ("roc_auc", 1),
    "accuracy": ("accuracy", 1),
    "f1": ("f1_weighted", 1),
    "log_loss": ("neg_log_loss", -1),
    "rmse": ("neg_root_mean_squared_error", -1),
    "mae": ("neg_mean_absolute_error", -1),
    "r2": ("r2", 1),
    "auc_ovr": ("roc_auc_ovr_weighted", 1),
}


@dataclass
class EvaluationResult:
    """Record paired-fold prediction metrics and actual work for one feature set.

    Attributes:
        value (float): Mean primary metric in its natural direction/units.
        metrics (dict[str, float]): Mean value for every requested metric.
        fold_values (list[float]): Primary metric in prepared split order.
        importance (dict[str, float]): Mean validation permutation importance by compiled column.
        directions (dict[str, str]): Descriptive Spearman direction for numeric derived features.
        duration (float): Evaluation wall-clock seconds; cache hits measure lookup time.
        model_fits (int): New fold model-fit attempts during this evaluation.
        transform_time (float): Seconds spent loading or constructing compiled fold matrices.
        training_time (float): Seconds spent fitting preprocessing plus estimator.
        prediction_time (float): Seconds scoring validation predictions across all metrics.
        explanation_time (float): Seconds computing permutation and direction evidence.
        cache_hit (bool): Whether a complete evaluation result was reused.
        feature_set_hash (str or None): Name-sensitive identity of the caller feature set.
    """

    value: float
    metrics: dict[str, float]
    fold_values: list[float]
    importance: dict[str, float] = field(default_factory=dict)
    directions: dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    model_fits: int = 0
    transform_time: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    explanation_time: float = 0.0
    cache_hit: bool = False
    feature_set_hash: str | None = None

    def to_dict(self):
        """Convert the result to a detached dataclass payload for persistence.

        Returns:
            dict: All result fields, recursively copied by dataclasses.asdict.
        """
        return asdict(self)


class CVEvaluator:
    """Evaluate controlled feature sets on fixed paired CV folds.

    Attributes:
        splits_ (list[tuple[ndarray, ndarray]]): Validated training/validation indices after prepare.
        dataset_hash_ (str): Prepared X, positional targets and groups identity used in cache keys.
        input_hash_ (str): X/y identity detecting replacement or in-place mutation.
        cache_namespace (str or None): Experiment namespace injected by FeatureStudy.
        on_model_fit (callable or None): Hook invoked immediately before each new fold fit.
        last_model_fits_ (int): Fit attempts charged by the most recent evaluate call.

    Notes:
        Baseline and candidates share splits. All learned preprocessing remains inside
        the training fold; validation rows are used only for scoring and explanation.
        See __init__ for estimator, metric and cache configuration.
    """

    def __init__(
        self,
        estimator=None,
        metric="auc",
        cv=5,
        metrics=None,
        random_state=42,
        n_jobs=1,
        importance_repeats=None,
        importance_max_samples=500,
        cache=None,
    ):
        """Configure paired-fold evaluation without fitting the estimator.

        Args:
            estimator (sklearn estimator or None): Explicit model, or a pinned TabPFN classifier/
                regressor chosen from metric when omitted.
            metric (str): Primary metric key in METRICS; scores retain their natural units.
            cv (int, splitter, iterable, or str): Fold count, sklearn splitter, index pairs, or "time".
            metrics (sequence[str] or None): Additional metric keys evaluated on the same fitted folds.
            random_state (int): Seed for reproducible splitting or sampling; does not seed arbitrary
                estimators.
            n_jobs (int): Parallel workers used for permutation importance, not fold training.
            importance_repeats (int or None): Permutations per column. None selects zero for the
                TabPFN adapter and two for other estimators; explicit zero disables explanation.
            importance_max_samples (int): Maximum validation rows per permutation-importance calculation.
            cache (ArtifactCache or None): Optional trusted memoization store for fold artifacts and
                results.

        Raises:
            ValueError: Metric names or explanation budgets are unsupported.
        """
        if metric not in METRICS or any(m not in METRICS for m in metrics or []):
            raise ValueError(f"Supported metrics: {list(METRICS)}")
        estimator = estimator if estimator is not None else default_estimator(metric, random_state)
        if importance_repeats is None:
            importance_repeats = 0 if isinstance(estimator, _TabPFNBase) else 2
        if isinstance(estimator, _TabPFNBase) and importance_repeats and n_jobs != 1:
            raise ValueError("TabPFN permutation importance requires n_jobs=1 to bound device memory")
        if importance_repeats < 0 or importance_max_samples < 1:
            raise ValueError("Invalid importance budget")
        self.cache = cache
        self.cache_namespace = None
        self.on_model_fit = None
        self.estimator = estimator
        self.metric = metric
        self.cv = cv
        self.metrics = list(dict.fromkeys([metric, *(metrics or [])]))
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_repeats = importance_repeats
        self.importance_max_samples = importance_max_samples

    def prepare(self, X, y, groups=None):
        """Validate row alignment and materialize a reusable split plan.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            y (array-like): One-dimensional targets aligned with X; never sent to the LLM.
            groups (array-like or None): Group labels aligned with X; groups must not cross CV folds.

        Returns:
            CVEvaluator: This evaluator with replaced splits and data fingerprints.

        Raises:
            ValueError: Targets/groups are misaligned, folds are invalid/overlapping, or fewer than two
                folds exist.

        Notes:
            Integer CV uses stratification for classifiers or GroupKFold when groups are
            supplied. Time CV assumes rows are already sorted. Custom iterable splits are
            consumed here; provide a reusable iterable if prepare will be called again.
        """
        if np.asarray(y).ndim != 1 or len(X) != len(y) or pd.isna(y).any():
            raise ValueError("X/y lengths must match and targets cannot be missing")
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("X/y indexes must align; positional targets may be a numpy array")
        if groups is not None and len(groups) != len(X):
            raise ValueError("groups must align with X")
        if isinstance(groups, pd.Series) and not X.index.equals(groups.index):
            raise ValueError("X/groups indexes must align")
        cv = self.cv
        if isinstance(cv, int):
            if groups is not None:
                cv = GroupKFold(cv)
            elif is_classifier(self.estimator):
                cv = StratifiedKFold(cv, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(cv, shuffle=True, random_state=self.random_state)
        elif isinstance(cv, str):
            if cv != "time":
                raise ValueError("String CV must be 'time'; rows must already be time ordered")
            cv = TimeSeriesSplit(5)
        splits = list(cv.split(X, y, groups)) if hasattr(cv, "split") else list(cv)
        if len(splits) < 2:
            raise ValueError("At least two folds are required")
        self.splits_ = []
        for train, valid in splits:
            train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
            if (
                len(train) == 0
                or len(valid) == 0
                or np.intersect1d(train, valid).size
                or min(train.min(), valid.min()) < 0
                or max(train.max(), valid.max()) >= len(X)
                or len(np.unique(train)) != len(train)
                or len(np.unique(valid)) != len(valid)
            ):
                raise ValueError("Invalid or overlapping train/validation indexes")
            if groups is not None:
                group_values = np.asarray(groups)
                if np.intersect1d(group_values[train], group_values[valid]).size:
                    raise ValueError("Groups overlap between training and validation")
            self.splits_.append((train, valid))
        self.dataset_hash_ = joblib.hash((X, np.asarray(y), groups))
        self.input_hash_ = joblib.hash((X, y))
        return self

    def evaluate(self, X, y, schema, features, model_params=None):
        """Score one controlled feature set on the prepared training/validation folds.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            y (array-like): One-dimensional targets aligned with X; never sent to the LLM.
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            features (FeatureSet or sequence[FeatureSpec]): Controlled derived features, excluding raw
                columns.
            model_params (dict or None): Estimator parameter overrides for this candidate.

        Returns:
            EvaluationResult: Mean/fold metrics, explanation evidence, timing and fit counts.

        Raises:
            ValueError: X/y changed since prepare, feature contracts fail, or a metric is non-finite.
            BudgetExceeded: The study model-fit hook refuses additional work.

        Notes:
            Automatically prepares on first use. For a new dataset, call prepare explicitly.
            Evaluation caches and fold matrices/models are name-sensitive because arbitrary
            estimators may inspect DataFrame column names. Estimator/scorer errors propagate
            to the study, which decides whether to record them as failed trials.
        """
        if not hasattr(self, "splits_"):
            self.prepare(X, y)
        elif joblib.hash((X, y)) != self.input_hash_:
            raise ValueError(
                "Evaluation data changed; call prepare(X, y, groups) before reusing this evaluator"
            )
        start = time.monotonic()
        self.last_model_fits_ = 0
        feature_set = FeatureSet.from_features(schema, features)
        features = feature_set.to_specs()
        y = np.asarray(y)
        estimator = clone(self.estimator).set_params(**(model_params or {}))
        key = {
            "experiment": self.cache_namespace,
            "dataset": self.dataset_hash_,
            "schema": content_hash(schema.model_dump()),
            "cv": joblib.hash(self.splits_),
            "features": feature_set.content_hash,
            "estimator": joblib.hash(estimator),
            "compiler": COMPILER_VERSION,
            "metrics": self.metrics,
            "explanation": [self.importance_repeats, self.importance_max_samples, self.random_state],
        }
        # Arbitrary DataFrame estimators may depend on column names, even for identical expressions.
        evaluation_key = key
        if self.cache:
            cached = self.cache.get("evaluation", evaluation_key)
            if cached is not None:
                result = EvaluationResult(**cached)
                result.feature_set_hash = feature_set.content_hash
                result.model_fits, result.transform_time, result.training_time = 0, 0.0, 0.0
                result.prediction_time = result.explanation_time = 0.0
                result.duration, result.cache_hit = time.monotonic() - start, True
                return result
        transform_time = training_time = prediction_time = explanation_time = 0.0
        scores = {metric: [] for metric in self.metrics}
        importance, correlations = {}, {}
        for fold, (train, valid) in enumerate(self.splits_, 1):
            logging.getLogger("featune").info(
                "CV fold %d/%d | train=%d validation=%d", fold, len(self.splits_), len(train), len(valid)
            )
            fold_key = {**key, "fold": fold}
            transform_start = time.monotonic()
            matrices = self.cache.get("matrix", fold_key) if self.cache else None
            if matrices is None:
                compiler = self.cache.get("fold_transform", fold_key) if self.cache else None
                if compiler is None:
                    # Learned transforms must never see the validation fold during fitting.
                    compiler = build_pipeline(schema, feature_set, clone(estimator))["features"].fit(
                        X.iloc[train]
                    )
                    if self.cache:
                        self.cache.put("fold_transform", fold_key, compiler)
                matrices = (compiler.transform(X.iloc[train]), compiler.transform(X.iloc[valid]))
                if self.cache:
                    self.cache.put("matrix", fold_key, matrices)
            training, engineered = matrices
            transform_time += time.monotonic() - transform_start
            remainder = self.cache.get("estimator", fold_key) if self.cache else None
            if remainder is None:
                # Matrices are already compiled; fit only preprocessing and the estimator here.
                remainder = Pipeline(build_pipeline(schema, feature_set, clone(estimator)).steps[1:])
                training_start = time.monotonic()
                # Charge before fit: an estimator failure must still consume the attempt budget.
                if self.on_model_fit:
                    self.on_model_fit()
                self.last_model_fits_ += 1
                remainder.fit(training, y[train])
                training_time += time.monotonic() - training_start
                if self.cache:
                    self.cache.put("estimator", fold_key, remainder)
            prediction_start = time.monotonic()
            for metric in self.metrics:
                scorer, sign = METRICS[metric]
                # Convert sklearn negative-loss scores back to the reported metric units.
                value = float(get_scorer(scorer)(remainder, engineered, y[valid]) * sign)
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite {metric}; check folds and target distribution")
                scores[metric].append(value)
            prediction_time += time.monotonic() - prediction_start
            logging.getLogger("featune").info(
                "CV fold %d/%d | %s=%.6f",
                fold,
                len(self.splits_),
                self.metric,
                scores[self.metric][-1],
            )
            explanation_start = time.monotonic()
            if self.importance_repeats:
                # Permute compiled columns independently; descendants are not recomputed.
                evidence = permutation_importance(
                    remainder,
                    engineered,
                    y[valid],
                    scoring=METRICS[self.metric][0],
                    n_repeats=self.importance_repeats,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_samples=min(self.importance_max_samples, len(valid)),
                )
                for name, value in zip(engineered.columns, evidence.importances_mean):
                    importance.setdefault(name, []).append(float(value))
                # Spearman association is descriptive, not a SHAP or causal estimate.
                target = pd.Series(y[valid]).reset_index(drop=True)
                if pd.api.types.is_numeric_dtype(target) and target.nunique() > 1:
                    for feature in features:
                        values = engineered[feature.name].reset_index(drop=True)
                        if pd.api.types.is_numeric_dtype(values) and values.nunique() > 1:
                            correlation = values.corr(target, method="spearman")
                            if pd.notna(correlation):
                                correlations.setdefault(feature.name, []).append(float(correlation))
            explanation_time += time.monotonic() - explanation_start
        result = EvaluationResult(
            value=float(np.mean(scores[self.metric])),
            metrics={name: float(np.mean(values)) for name, values in scores.items()},
            fold_values=scores[self.metric],
            importance={name: float(np.mean(values)) for name, values in importance.items()},
            directions={
                name: (
                    "increasing"
                    if np.mean(values) > 0.05
                    else "decreasing"
                    if np.mean(values) < -0.05
                    else "unknown"
                )
                for name, values in correlations.items()
            },
            duration=time.monotonic() - start,
            model_fits=self.last_model_fits_,
            transform_time=transform_time,
            training_time=training_time,
            prediction_time=prediction_time,
            explanation_time=explanation_time,
            feature_set_hash=feature_set.content_hash,
        )
        if self.cache:
            self.cache.put("evaluation", evaluation_key, result.to_dict())
        return result
