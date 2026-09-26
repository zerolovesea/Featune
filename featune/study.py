# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Feature trials, bounded search, restart safety and fitted export.

Coordinates proposal, validation, evaluation, accounting and durable recovery.
Callers supply training data only; independent test evaluation belongs outside Study.

Created:
    2026-09-21
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .budget import SearchBudget
from .cache import ArtifactCache
from .compiler import FeatureCompiler, build_pipeline
from .evaluation import METRICS, CVEvaluator
from .fingerprint import ExperimentFingerprint
from .ir import FeatureSet, content_hash
from .llm import BudgetExceeded, LLMError
from .samplers import (
    AutonomousSampler,
    HybridSampler,
    LLMSampler,
    RandomSampler,
    SearchContext,
    validate_params,
)
from .schema import DatasetSchema, FeatureSpec, validate_features
from .storage import Storage, dumps

logger = logging.getLogger("featune")


@dataclass
class Trial:
    """Persist the proposal, lineage, evaluation and resource evidence for one attempt.

    Attributes:
        number (int): Zero for baseline; positive, never-reused candidate attempt identifier.
        state (str): RUNNING, COMPLETE, FAIL, DUPLICATE, INTERRUPTED or BUDGET_EXCEEDED.
        features (list[dict]): Full candidate FeatureSpecs, including retained parent features.
        model_params (dict): Effective candidate estimator overrides.
        parent (int): Parent trial number.
        value (float or None): Mean primary metric for completed/reused evaluation.
        delta (float or None): Candidate minus baseline metric, without direction normalization.
        parent_delta (float or None): Candidate minus parent metric; negative improves a loss.
        metrics (dict): All evaluated mean metrics.
        fold_values (list[float]): Primary metric in paired-fold order.
        importance (dict): Validation permutation importance by compiled column.
        hypotheses (dict): Per-feature joint-evidence status and descriptive direction.
        tokens (dict): Input/output/request/unknown billing deltas for this attempt.
        duration (float): Attempt wall-clock seconds.
        error (str or None): Sanitized failure/stop reason.
        signature (str or None): Name-sensitive feature-set plus model-parameter identity.
        stage (str): proposal or evaluation, used for recovery and failure accounting.
        parent_set (dict): Serialized parent FeatureSet.
        proposal (dict): Parsed accepted proposal extension.
        candidate_set (dict): Serialized full candidate FeatureSet.
        feature_count (int): Number of derived candidate columns.
        model_fits (int): Fit attempts charged to this trial.
        transform_time (float): Seconds compiling/loading fold matrices.
        training_time (float): Seconds fitting fold preprocessing and estimators.
        prediction_time (float): Seconds scoring validation predictions.
        explanation_time (float): Seconds computing validation explanations.
        cache_hit (bool): Complete evaluation was reused rather than retrained.
        valid_proposal (bool): Proposal passed study validation.
        api_cost (float or None): Known estimated cost delta; None means pricing is unavailable.
        elapsed (float): Cumulative active-study seconds at trial finalization.
        context_metadata (dict): Bounded field selection, prompt hashes, request usage and sanitized
            failures.
    """

    number: int
    state: str = "RUNNING"
    features: list[dict] = field(default_factory=list)
    model_params: dict = field(default_factory=dict)
    parent: int = 0
    value: float | None = None
    delta: float | None = None
    parent_delta: float | None = None
    metrics: dict = field(default_factory=dict)
    fold_values: list[float] = field(default_factory=list)
    importance: dict = field(default_factory=dict)
    hypotheses: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    duration: float = 0.0
    error: str | None = None
    signature: str | None = None
    stage: str = "proposal"
    parent_set: dict = field(default_factory=dict)
    proposal: dict = field(default_factory=dict)
    candidate_set: dict = field(default_factory=dict)
    feature_count: int = 0
    model_fits: int = 0
    transform_time: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    explanation_time: float = 0.0
    cache_hit: bool = False
    valid_proposal: bool = False
    api_cost: float | None = None
    elapsed: float = 0.0
    context_metadata: dict = field(default_factory=dict)

    def to_dict(self):
        """Serialize all trial fields for SQLite, callbacks and tabular inspection.

        Returns:
            dict: Recursively copied dataclass payload; no live estimator or raw data.
        """
        return asdict(self)


class FeatureStudy:
    """Coordinate controlled feature search, paired evaluation, budgets and durable history.

    Attributes:
        trials (list[Trial]): Baseline and candidate records in attempt order.
        sampler (BaseFeatureSampler): Proposal policy whose configuration/state is fingerprinted.
        storage (Storage or None): Optional per-study database and artifact directory.
        budget (SearchBudget): Cumulative search ceilings.
        stop_requested (bool): Callback-requested stop checked between trials.
        stop_reason (str or None): Most recent termination reason.
        fingerprint (ExperimentFingerprint): Identity available after loading/preparing a study.
        pipeline_ (sklearn.pipeline.Pipeline): Best refitted model, present only when refit succeeds
            or is cached.

    Notes:
        Use optimize with outer-training data only. Properties such as best_trial are
        unavailable until a baseline/candidate completes. Concurrent writes to the same
        study are rejected by Storage.lock; instances themselves are not thread-safe.
        See __init__ for search configuration and optimize for execution semantics.
    """

    def __init__(
        self,
        metric="auc",
        direction=None,
        sampler=None,
        storage=None,
        study_name="default",
        search_strategy="greedy",
        beam_width=3,
        max_features=32,
        param_space=None,
        random_state=42,
        budget=None,
        joint_optimization=False,
        include_statistics=False,
        cache=True,
    ):
        """Validate search settings and load any existing per-study records.

        Args:
            metric (str): Primary metric key in METRICS; scores retain their natural units.
            direction (str or None): Inferred from metric when omitted; explicit direction must match.
            sampler (BaseFeatureSampler or None): Proposal policy; None selects the study default.
            storage (str, Path, or None): Root for a study_name subdirectory; None keeps history in
                memory.
            study_name (str): Single directory name identifying the study; path traversal is disallowed.
            search_strategy (str): independent starts from baseline; greedy/beam extend ranked parents.
            beam_width (int): Maximum retained parents expanded per frozen beam generation.
            max_features (int): Maximum derived-feature count; at most 64 in study/DSL validation.
            param_space (dict or None): Nonempty finite approved estimator choices; requires joint
                optimization.
            random_state (int): Seed for reproducible splitting or sampling; does not seed arbitrary
                estimators.
            budget (SearchBudget or dict or None): Cumulative trial, fit, time, token and cost limits.
            joint_optimization (bool): Allow model-parameter proposals in addition to features; disabled
                by default.
            include_statistics (bool): Compute selected-column, target-free training summaries for LLM
                context.
            cache (bool): Enable numerical artifact caches; duplicate trial metrics are still reused when
                False.

        Raises:
            ValueError: Direction, strategy, budgets, study name or parameter-space settings are invalid.

        Notes:
            Creating disk storage initializes SQLite tables. Loading does not refit a model
            or reconstruct training data; numerical compatibility is checked by optimize.
        """
        if metric not in METRICS:
            raise ValueError(f"Supported metrics: {list(METRICS)}")
        expected_direction = "maximize" if METRICS[metric][1] == 1 else "minimize"
        direction = expected_direction if direction is None else direction
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be maximize or minimize")
        if direction != expected_direction:
            raise ValueError(f"{metric} requires direction='{expected_direction}'")
        if search_strategy not in {"independent", "greedy", "beam", "autonomous"}:
            raise ValueError("search_strategy must be independent, greedy, beam or autonomous")
        if beam_width < 1 or max_features < 1 or max_features > 64:
            raise ValueError("Invalid feature/beam budgets")
        if not study_name or Path(study_name).name != study_name or study_name in {".", ".."}:
            raise ValueError("study_name must be a single directory name")
        self.metric, self.direction = metric, direction
        self.sampler = sampler or (
            AutonomousSampler(numeric_sampler=RandomSampler(random_state))
            if search_strategy == "autonomous"
            else RandomSampler(random_state)
        )
        self.study_name, self.search_strategy = study_name, search_strategy
        self.beam_width, self.max_features = beam_width, max_features
        self.budget = SearchBudget.model_validate(budget or {})
        self.joint_optimization = joint_optimization
        self.include_statistics = include_statistics
        self.cache_enabled = cache
        self.param_space = param_space or {}
        if self.param_space and not joint_optimization:
            raise ValueError("Set joint_optimization=True to enable param_space")
        for name, choices in self.param_space.items():
            if not isinstance(name, str) or not isinstance(choices, list) or not choices:
                raise ValueError("param_space maps estimator parameter names to nonempty lists")
        dumps(self.param_space)
        self.random_state = random_state
        self.storage = Storage(Path(storage) / study_name) if storage is not None else None
        self.trials: list[Trial] = []
        self.stop_requested = False
        self.stop_reason = None
        self._metadata = {}
        self._consumption = {
            "trials": 0,
            "model_fits": 0,
            "wall_time": 0.0,
            "valid_proposals": 0,
            "llm_tokens": 0,
            "llm_cost": 0.0,
        }
        self._active_trial = None
        if self.storage:
            self._reload()

    def _reload(self):
        """Replace in-memory history and counters with persisted records.

        Returns:
            None: Updates metadata, trials, consumption and any saved fingerprint.

        Notes:
            Requires configured storage; called under the writer lock before optimization.
        """
        self._metadata, history = self.storage.load()
        self.trials = [Trial(**trial) for trial in history]
        self._consumption = self._metadata.get("consumption", self._consumption)
        self.stop_reason = self._metadata.get("stop_reason")
        if "experiment_fingerprint" in self._metadata:
            self.fingerprint = ExperimentFingerprint.model_validate(self._metadata["experiment_fingerprint"])

    @property
    def best_trial(self):
        """Select the best completed trial, including the original baseline.

        Returns:
            Trial: Highest/lowest value for the configured direction; ties keep earlier records.

        Raises:
            ValueError: No trial has state COMPLETE.
        """
        completed = [trial for trial in self.trials if trial.state == "COMPLETE"]
        if not completed:
            raise ValueError("No completed trials")
        return sorted(completed, key=lambda trial: trial.value, reverse=self.direction == "maximize")[0]

    @property
    def best_value(self):
        """Read the primary metric of the best completed trial.

        Returns:
            float: Inner-CV value, not an unbiased outer-test estimate.

        Raises:
            ValueError: No completed trial exists.
        """
        return self.best_trial.value

    @property
    def best_features(self):
        """Reconstruct controlled specs for the best completed candidate.

        Returns:
            list[FeatureSpec]: Validated derived features; may be empty when baseline wins.

        Raises:
            ValueError: No completed trial exists or a stored spec is invalid.
        """
        return [FeatureSpec.model_validate(feature) for feature in self.best_trial.features]

    @property
    def baseline_trial(self):
        """Locate the completed unengineered baseline record.

        Returns:
            Trial: Completed trial zero.

        Raises:
            StopIteration: The baseline has not completed.
        """
        return next(trial for trial in self.trials if trial.number == 0 and trial.state == "COMPLETE")

    def _client(self):
        """Find the built-in LLM client behind an LLM or hybrid sampler.

        Returns:
            LLMClient or None: Client used for accounting; None for non-LLM policies.
        """
        sampler = (
            self.sampler.llm_sampler
            if isinstance(self.sampler, (HybridSampler, AutonomousSampler))
            else self.sampler
        )
        return sampler.client if isinstance(sampler, LLMSampler) else None

    @staticmethod
    def _screen_candidate(X, schema, evaluator, candidate, new_names):
        """Reject a new column only when every training fold finds it constant."""
        if not new_names:
            return None
        for train, _ in evaluator.splits_:
            compiled = FeatureCompiler(schema, candidate).fit_transform(X.iloc[train])
            for name in new_names:
                values = compiled[name]
                if values.nunique(dropna=False) > 1:
                    return None
        return "constant_in_every_training_fold"

    def _usage(self):
        """Read a detached snapshot of known LLM usage counters.

        Returns:
            dict[str, int]: TokenUsage counters, or zero counters for a non-LLM sampler.
        """
        client = self._client()
        return (
            client.usage.to_dict()
            if client
            else {"input_tokens": 0, "output_tokens": 0, "requests": 0, "unknown_requests": 0}
        )

    def _save(self, trial=None, **metadata):
        """Snapshot sampler state, usage and budgets alongside an optional trial.

        Args:
            trial (Trial or None): Record to persist atomically with the metadata.
            metadata (Any): Additional JSON-serializable metadata updates.

        Returns:
            None: Updates metadata and commits to storage when configured.

        Notes:
            Storage/serialization failures propagate; treating a failed write as a failed
            proposal could hide lost recovery state.
        """
        self._metadata.update(metadata)
        self._metadata["sampler_state"] = self.sampler.state_dict()
        self._metadata["consumption"] = self.consumption
        self._metadata["budget"] = self.budget.model_dump()
        if self.storage:
            self.storage.save(trial.to_dict() if trial else None, **self._metadata)

    def stop(self):
        """Request termination after the current attempt finishes.

        Returns:
            None: Sets stop_requested for the next trial-boundary check.

        Notes:
            Suitable for callbacks; does not interrupt active model training or HTTP calls.
        """
        self.stop_requested = True

    @property
    def consumption(self):
        """Report cumulative work and the active client's billing counters.

        Returns:
            dict: Trial/fit/time/valid-proposal totals plus known tokens, cost and unknown requests.

        Notes:
            LLM totals are read from client.usage, which optimize restores from persistence.
            Wall time is refreshed at work boundaries; this property is not a live timer.
        """
        usage = dict(self._consumption)
        client = self._client()
        if client:
            usage["llm_tokens"] = client.usage.total_tokens
            usage["llm_cost"] = client.cost
            usage["unknown_requests"] = client.usage.unknown_requests
        return usage

    def _charge_fit(self):
        """Reserve one model fit and persist its charge before training starts.

        Returns:
            None: Increments total and active-trial fit counts.

        Raises:
            BudgetExceeded: The fit count or active wall-time ceiling is already exhausted.

        Notes:
            Failed/interrupted fits still consume a reservation. Requires optimize timer
            initialization; called through the evaluator hook and before final refit.
        """
        limit = self.budget.max_model_fits
        if limit is not None and self._consumption["model_fits"] >= limit:
            raise BudgetExceeded("Model fit budget exhausted")
        if self.budget.max_wall_time is not None and self._wall_time() >= self.budget.max_wall_time:
            raise BudgetExceeded("Wall time budget exhausted")
        self._consumption["model_fits"] += 1
        if self._active_trial:
            self._active_trial.model_fits += 1
        self._save(self._active_trial)

    def _wall_time(self):
        """Measure accumulated active optimize time including this invocation.

        Returns:
            float: Prior recorded seconds plus current monotonic elapsed seconds.
        """
        return self._previous_wall + time.monotonic() - self._started

    def _remember_set(self, feature_set):
        """Memoize canonical IR nodes and the enclosing FeatureSet if caching is enabled.

        Args:
            feature_set (FeatureSet): Validated canonical derived-feature collection.

        Returns:
            FeatureSet: The same input collection; its numerical values are not evaluated.
        """
        if self._cache is not None:
            if self._cache.get("feature_set", feature_set.content_hash) is None:
                self._cache.put("feature_set", feature_set.content_hash, feature_set.model_dump())
            for feature in feature_set.features:
                if self._cache.get("ir", feature.content_hash) is None:
                    self._cache.put("ir", feature.content_hash, feature.model_dump())
        return feature_set

    def optimize(
        self,
        X,
        y,
        schema,
        estimator=None,
        n_trials=30,
        evaluator=None,
        groups=None,
        timeout=None,
        patience=None,
        callbacks=(),
        catch=(ValueError, RuntimeError),
        refit=True,
        allow_incompatible_resume=False,
    ):
        """Run or resume bounded feature search on the supplied training dataset.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            y (array-like): One-dimensional targets aligned with X; never sent to the LLM.
            schema (DatasetSchema or dict): Validated field types, semantics, exclusions and task objective.
            estimator (sklearn estimator or None): Used when evaluator is omitted; None selects
                the pinned TabPFN preset matching the study metric.
            n_trials (int): Additional candidate attempts for this call; excludes the baseline.
            evaluator (CVEvaluator or None): Custom evaluator whose primary metric must match the study.
            groups (array-like or None): Group labels aligned with X; groups must not cross CV folds.
            timeout (float or None): Positive per-call wall-clock seconds, checked between trials.
            patience (int or None): Consecutive non-improving attempts allowed within this optimize call.
            callbacks (iterable[callable]): Functions called as callback(study, trial) after each
                finalized attempt.
            catch (tuple[type[Exception]]): Candidate evaluation errors converted to FAIL records.
            refit (bool): Whether to fit/export the best candidate on all supplied training rows if budget
                permits.
            allow_incompatible_resume (bool): Explicitly archive/reset incompatible history instead of
                rejecting resume.

        Returns:
            FeatureStudy: This study, including finalized records and an optional fitted pipeline_.

        Raises:
            ValueError: Input/configuration is invalid or resume fingerprint is incompatible.
            RuntimeError: A different writer owns this study, or an uncaught execution failure occurs.

        Notes:
            Materializes paired folds, checks fingerprints under the storage lock, records
            interrupted attempts, evaluates a baseline, then proposes/evaluates extensions.
            Candidate errors in catch and LLMError become failed trials; baseline and
            persistence errors propagate. BudgetExceeded becomes a normal stop. Callbacks
            run after records are saved. refit may be skipped if fit/time budget is exhausted.
            No outer-test data or scores should be supplied. allow_incompatible_resume
            archives the old experiment and resets counters instead of mixing results.
        """
        if n_trials < 0 or (timeout is not None and timeout <= 0) or (patience is not None and patience < 1):
            raise ValueError("Invalid optimization budget")
        if patience is None and self.search_strategy == "autonomous":
            patience = 8
        schema = DatasetSchema.model_validate(schema)
        schema.validate_frame(X)
        evaluator = evaluator or CVEvaluator(estimator, metric=self.metric, random_state=self.random_state)
        if evaluator.estimator is None or evaluator.metric != self.metric:
            raise ValueError("Provide an estimator and an evaluator with the study metric")
        for name, choices in self.param_space.items():
            for choice in choices:
                clone(evaluator.estimator).set_params(**{name: choice})
        evaluator.prepare(X, y, groups)
        configuration = {
            "format_version": 2,
            "metric": self.metric,
            "direction": self.direction,
            "schema": schema.model_dump(),
            "search_strategy": self.search_strategy,
            "beam_width": self.beam_width,
            "max_features": self.max_features,
            "param_space": self.param_space,
            "random_state": self.random_state,
            "sampler": self.sampler.configuration(),
            "joint_optimization": self.joint_optimization,
            "include_statistics": self.include_statistics,
            "cache": self.cache_enabled,
        }
        fingerprint = ExperimentFingerprint.capture(
            X, np.asarray(y), groups, schema, evaluator, self.sampler, self.random_state, configuration
        )
        identity = fingerprint.compatibility_hash
        client = self._client()
        if client:
            if self.budget.max_llm_tokens is not None:
                client.token_budget = self.budget.max_llm_tokens
            if self.budget.max_llm_cost is not None:
                if client.cost is None:
                    raise ValueError("max_llm_cost requires explicit LLM input/output token prices")
                client.cost_budget = self.budget.max_llm_cost
        # Reload under the writer lock so another process cannot leave us with stale history.
        with self.storage.lock() if self.storage else nullcontext():
            if self.storage:
                self._reload()
            if self._metadata.get("fingerprint", identity) != identity:
                if not allow_incompatible_resume:
                    raise ValueError(
                        "Resume rejected: data, schema, estimator, folds, versions or configuration changed"
                    )
                if self.storage:
                    self.storage.archive_and_reset(self._metadata, [t.to_dict() for t in self.trials])
                self.trials, self._metadata = [], {}
                self.sampler.load_state_dict({})
                self._consumption = {
                    "trials": 0,
                    "model_fits": 0,
                    "wall_time": 0.0,
                    "valid_proposals": 0,
                    "llm_tokens": 0,
                    "llm_cost": 0.0,
                }
            self.sampler.load_state_dict(self._metadata.get("sampler_state", self.sampler.state_dict()))
            self.fingerprint = fingerprint
            directory = self.storage.directory / "cache" if self.storage else None
            previous_cache = getattr(self, "_cache", None)
            self._cache = (
                (
                    previous_cache
                    if previous_cache is not None and previous_cache.namespace == identity
                    else ArtifactCache(directory, identity)
                )
                if self.cache_enabled
                else None
            )
            evaluator.cache, evaluator.cache_namespace = self._cache, identity
            evaluator.on_model_fit = self._charge_fit
            self._started = time.monotonic()
            self._previous_wall = self._consumption["wall_time"]
            handler = None
            if self.storage:
                handler = logging.FileHandler(self.storage.directory / "search.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                logger.addHandler(handler)
            try:
                self._save(
                    configuration=configuration,
                    fingerprint=identity,
                    experiment_fingerprint=fingerprint.model_dump(),
                )
                for trial in self.trials:
                    if trial.state == "RUNNING":
                        trial.state, trial.error = (
                            "INTERRUPTED",
                            "Previous process stopped before trial completion",
                        )
                        # A stopped proposal may have reached the provider without returning usage.
                        if client and trial.number and trial.stage == "proposal":
                            client.usage.unknown_requests += 1
                        self._save(trial)
                self._X, self._y, self.schema, self.evaluator = X, y, schema, evaluator
                # A new search invalidates any previously fitted export, even when refit=False.
                self.__dict__.pop("pipeline_", None)
                self.stop_requested, self.stop_reason = False, None
                statistics = {}

                def selected_statistics(names):
                    """Compute missing selected-column summaries once within the current optimize call.

                    Args:
                        names (list[str]): Selected usable raw-column names requiring statistical summaries.

                    Returns:
                        dict: Summaries for exactly the requested names.

                    Notes:
                        Mutates the closure-local statistics cache and never receives target values.
                    """
                    missing = [name for name in names if name not in statistics]
                    statistics.update(self._column_statistics(X, schema, missing))
                    return {name: statistics[name] for name in names}

                if not any(t.number == 0 and t.state == "COMPLETE" for t in self.trials):
                    if (
                        self.budget.max_model_fits is not None
                        and self._consumption["model_fits"] + len(evaluator.splits_)
                        > self.budget.max_model_fits
                    ):
                        self.stop_reason = "max_model_fits"
                        return self
                    self.trials = [t for t in self.trials if t.number != 0]
                    baseline = Trial(number=0, stage="evaluation", api_cost=0.0)
                    baseline_set = self._remember_set(FeatureSet.from_features(schema, []))
                    baseline.candidate_set = baseline_set.model_dump()
                    baseline.signature = content_hash([baseline_set.content_hash, {}])
                    self.trials.append(baseline)
                    self._active_trial = baseline
                    # Persist RUNNING before fitting so process failures remain recoverable.
                    self._save(baseline)
                    logger.info("Baseline | fitting %d folds", len(evaluator.splits_))
                    try:
                        self._complete(baseline, evaluator.evaluate(X, y, schema, baseline_set), None)
                        logger.info("Baseline | COMPLETE | %s=%.6f", self.metric, baseline.value)
                    except BudgetExceeded as error:
                        baseline.state, baseline.error = "BUDGET_EXCEEDED", str(error)
                        self.stop_reason = "budget"
                        return self
                    finally:
                        baseline.elapsed = self._wall_time()
                        self._save(baseline)
                        self._active_trial = None
                stale = 0
                for _ in range(n_trials):
                    self._consumption["wall_time"] = self._wall_time()
                    exhausted = self.budget.exhausted(self.consumption)
                    if (
                        self.stop_requested
                        or exhausted
                        or (timeout is not None and time.monotonic() - self._started >= timeout)
                    ):
                        self.stop_reason = "callback" if self.stop_requested else exhausted or "timeout"
                        break
                    number = max(t.number for t in self.trials) + 1
                    parent = self._select_parent(number)
                    parent_set = FeatureSet.model_validate(parent.candidate_set)
                    parent_features = parent_set.to_specs()
                    trial = Trial(number=number, parent=parent.number, parent_set=parent_set.model_dump())
                    self.trials.append(trial)
                    self._active_trial = trial
                    self._consumption["trials"] += 1
                    # Record the attempt before external work can consume tokens or model fits.
                    self._save(trial)
                    usage_before, cost_before = self._usage(), client.cost if client else 0.0
                    trial_start, previous_best = time.monotonic(), self.best_value
                    logger.info("Trial %d | parent=%d | proposing features", number, parent.number)
                    try:
                        parent_estimator = clone(evaluator.estimator).set_params(**parent.model_params)
                        proposal = self.sampler.sample(
                            SearchContext(
                                schema,
                                number,
                                parent_features,
                                [t.to_dict() for t in self.trials[:-1]],
                                self.metric,
                                self.direction,
                                self.param_space,
                                parent.model_params,
                                self.max_features - len(parent_features),
                                estimator={
                                    "type": f"{type(parent_estimator).__module__}."
                                    f"{type(parent_estimator).__name__}",
                                    "model_version": getattr(parent_estimator, "model_version", None),
                                },
                                statistics_provider=selected_statistics if self.include_statistics else None,
                                context_metadata=trial.context_metadata,
                                current_feature_set=parent_set.model_dump(),
                                remaining_budget=self.budget.remaining(self.consumption),
                                parent_trial=parent.number,
                            )
                        )
                        if proposal.model_params and not self.joint_optimization:
                            raise ValueError("Joint model optimization is disabled")
                        if not proposal.features and not proposal.remove and not proposal.model_params:
                            raise ValueError("Empty proposal")
                        if len(set(proposal.remove)) != len(proposal.remove) or not set(proposal.remove) <= {
                            feature.name for feature in parent_features
                        }:
                            raise ValueError("Removal must name distinct parent features")
                        if proposal.remove:
                            trial.context_metadata.setdefault("removal_reason", "sampler_proposed_removal")
                        if self.search_strategy == "autonomous" and len(proposal.features) > 1:
                            raise ValueError("Autonomous search evaluates one new feature per trial")
                        features = [
                            f for f in parent_features if f.name not in proposal.remove
                        ] + proposal.features
                        validate_features(schema, features, self.max_features)
                        candidate = self._remember_set(
                            FeatureSet.from_features(
                                schema, features, source_trial=number, origins=parent_set
                            )
                        )
                        params = {**parent.model_params, **proposal.model_params}
                        validate_params(params, self.param_space)
                        # Valid proposals count even when an equivalent candidate reuses cached evidence.
                        trial.valid_proposal = True
                        self._consumption["valid_proposals"] += 1
                        trial.proposal, trial.candidate_set = proposal.model_dump(), candidate.model_dump()
                        trial.features, trial.model_params = (
                            [f.model_dump() for f in candidate.to_specs()],
                            params,
                        )
                        trial.feature_count = len(candidate.features)
                        trial.signature = content_hash([candidate.content_hash, params])
                        trial.stage = "evaluation"
                        trial.tokens = {
                            key: value - usage_before[key] for key, value in self._usage().items()
                        }
                        self._save(trial)
                        source = trial.context_metadata.get("sampler", type(self.sampler).__name__)
                        logger.info(
                            "Trial %d | proposal | sampler=%s parent_features=%s model_params=%s new_features=%d removed=%s removal_reason=%s",
                            number,
                            source,
                            dumps([feature.name for feature in parent_features]),
                            dumps(params),
                            len(proposal.features),
                            dumps(proposal.remove),
                            trial.context_metadata.get("removal_reason"),
                        )
                        for feature in proposal.features:
                            logger.info(
                                "Trial %d | hypothesis_proposed | %s",
                                number,
                                dumps(
                                    {
                                        "name": feature.name,
                                        "op": feature.op,
                                        "inputs": feature.inputs,
                                        "params": feature.params,
                                        **feature.hypothesis.model_dump(),
                                    }
                                ),
                            )
                        duplicate = next(
                            (
                                t
                                for t in self.trials[:-1]
                                if t.signature == trial.signature and t.state == "COMPLETE"
                            ),
                            None,
                        )
                        if duplicate:
                            # The durable trial record is also an evaluation-result cache when heavy caching is off.
                            from .evaluation import EvaluationResult

                            result = EvaluationResult(
                                value=duplicate.value,
                                metrics=duplicate.metrics,
                                fold_values=duplicate.fold_values,
                                importance=duplicate.importance,
                                directions={
                                    name: h["observed_direction"] for name, h in duplicate.hypotheses.items()
                                },
                                cache_hit=True,
                                feature_set_hash=candidate.content_hash,
                            )
                            self._complete(trial, result, parent)
                            trial.state = "DUPLICATE"
                        elif self.search_strategy == "autonomous" and (
                            reason := self._screen_candidate(
                                X, schema, evaluator, candidate, [f.name for f in proposal.features]
                            )
                        ):
                            trial.state, trial.error = "SCREENED_OUT", reason
                            trial.context_metadata["screening"] = reason
                            logger.info("Trial %d | screened_out | reason=%s", number, reason)
                        else:
                            logger.info(
                                "Trial %d | evaluating %d features over %d folds",
                                number,
                                len(features),
                                len(evaluator.splits_),
                            )
                            self._complete(trial, evaluator.evaluate(X, y, schema, candidate, params), parent)
                    except BudgetExceeded as error:
                        trial.state, trial.error = "BUDGET_EXCEEDED", str(error)
                        self.stop_reason = "budget"
                    except (LLMError, *catch) as error:
                        trial.state = "FAIL"
                        trial.error = (
                            str(error)
                            if isinstance(error, LLMError)
                            else (f"{type(error).__name__}: proposal validation or model evaluation failed")
                        )
                        logger.warning("Trial %d | %s", number, trial.error)
                    finally:
                        # Failed validation/evaluation still consumes time and may incur API charges.
                        trial.duration, trial.elapsed = time.monotonic() - trial_start, self._wall_time()
                        self._consumption["wall_time"] = trial.elapsed
                        trial.tokens = {
                            key: value - usage_before[key] for key, value in self._usage().items()
                        }
                        trial.api_cost = (
                            client.cost - cost_before
                            if client and client.cost is not None
                            else (None if client else 0.0)
                        )
                        self._save(trial)
                        self._active_trial = None
                    for feature in trial.proposal.get("features", []):
                        evidence = trial.hypotheses.get(feature["name"])
                        logger.info(
                            "Trial %d | hypothesis_assessed | %s",
                            number,
                            dumps(
                                {
                                    "name": feature["name"],
                                    "status": evidence["status"] if evidence else "not_evaluated",
                                    "reason": evidence["reason"] if evidence else trial.error,
                                    "observed_direction": evidence["observed_direction"]
                                    if evidence
                                    else None,
                                    "importance": evidence["importance"] if evidence else None,
                                    "positive_fold_fraction": (
                                        evidence["positive_fold_fraction"] if evidence else None
                                    ),
                                    "parent_delta": trial.parent_delta,
                                    "cache_hit": trial.cache_hit,
                                }
                            ),
                        )
                    if trial.state == "COMPLETE":
                        decision = "selected_best" if self.best_trial.number == number else "kept_not_best"
                    elif trial.state == "DUPLICATE":
                        decision = "reused_prior_evaluation"
                    else:
                        decision = "not_selected"
                    logger.info(
                        "Trial %d | %s | decision=%s value=%s parent_delta=%s best=%.6f "
                        "best_trial=%d | tokens=%d+%d fits=%d cost=%s | %.2fs",
                        number,
                        trial.state,
                        decision,
                        trial.value,
                        trial.parent_delta,
                        self.best_value,
                        self.best_trial.number,
                        trial.tokens["input_tokens"],
                        trial.tokens["output_tokens"],
                        trial.model_fits,
                        trial.api_cost,
                        trial.duration,
                    )
                    for callback in callbacks:
                        callback(self, trial)
                    improved = (
                        self.best_value > previous_best
                        if self.direction == "maximize"
                        else self.best_value < previous_best
                    )
                    stale = 0 if improved else stale + 1
                    if self.stop_reason == "budget":
                        break
                    if patience is not None and stale >= patience:
                        self.stop_reason = "patience"
                        break
                self.stop_reason = self.stop_reason or "n_trials"
                if refit:
                    refit_key = {"experiment": identity, "refit": self.best_trial.signature}
                    self.pipeline_ = self._cache.get("estimator", refit_key) if self._cache else None
                    if self.pipeline_ is None:
                        try:
                            self._charge_fit()
                        except BudgetExceeded:
                            # An unfitted placeholder must not appear as an exportable pipeline.
                            self.__dict__.pop("pipeline_", None)
                            logger.info("Refit skipped: no remaining model-fit or wall-time budget")
                        else:
                            logger.info(
                                "Refitting best trial %d on all supplied training rows",
                                self.best_trial.number,
                            )
                            pipeline = build_pipeline(
                                schema,
                                self.best_features,
                                clone(evaluator.estimator).set_params(**self.best_trial.model_params),
                            )
                            pipeline.fit(X, y)
                            self.pipeline_ = pipeline
                            if self._cache:
                                self._cache.put("estimator", refit_key, pipeline)
            finally:
                self._active_trial = None
                evaluator.on_model_fit = None
                self._consumption["wall_time"] = self._wall_time()
                self._save(stop_reason=self.stop_reason)
                if handler:
                    logger.removeHandler(handler)
                    handler.close()
        return self

    @staticmethod
    def _column_statistics(X, schema, names=None):
        """Summarize selected training columns without raw category labels or target values.

        Args:
            X (pandas.DataFrame): Input feature rows with unique columns and schema-compatible types.
            schema (DatasetSchema or dict): Validated field types, semantics, exclusions and task objective.
            names (list[str] or None): Limit computation to these fields; None summarizes every usable
                field.

        Returns:
            dict[str, dict]: Dtype, missingness/cardinality and optional numeric/frequency summaries.

        Notes:
            Numeric moments/quantiles require at least ten nonmissing observations.
            Categorical top frequencies contain proportions only, never category labels.
        """
        result = {}
        for name, column in schema.usable.items():
            if names is not None and name not in names:
                continue
            values = X[name]
            summary = {
                "dtype": column.dtype,
                "missing_rate": float(values.isna().mean()),
                "cardinality": int(values.nunique()),
            }
            if column.dtype == "numeric" and values.notna().sum() >= 10:
                finite = values.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite):
                    summary.update(
                        {
                            key: float(value)
                            for key, value in {
                                "min": finite.min(),
                                "max": finite.max(),
                                "mean": finite.mean(),
                                "std": finite.std(ddof=0),
                            }.items()
                        }
                    )
                    summary["quantiles"] = {str(q): float(finite.quantile(q)) for q in [0.1, 0.5, 0.9]}
            elif column.dtype == "categorical":
                # Frequencies disclose no raw category labels or individual rows.
                summary["top_category_frequencies"] = values.value_counts(normalize=True).head(5).tolist()
            result[name] = summary
        return result

    def _select_parent(self, number):
        """Choose the next independent, greedy or frozen-frontier beam parent.

        Args:
            number (int): Trial index used to choose the parent or deterministic exploration seed.

        Returns:
            Trial: Eligible completed parent, falling back to baseline when necessary.

        Notes:
            Beam freezes candidates at the generation cutoff so each retained parent gets
            an expansion even if an earlier child improves the current best score.
        """
        if self.search_strategy == "independent":
            return self.baseline_trial
        # Freeze the frontier for a generation so every retained parent gets an expansion.
        cutoff = (
            ((number - 1) // self.beam_width) * self.beam_width
            if self.search_strategy == "beam"
            else number - 1
        )

        def expandable(trial):
            if len(trial.features) < self.max_features:
                return True
            if self.search_strategy != "autonomous":
                return False
            used = {name for feature in trial.features for name in feature["inputs"]}
            leaves = {feature["name"] for feature in trial.features} - used
            tried = {
                name
                for attempt in self.trials
                if attempt.parent == trial.number
                for name in attempt.proposal.get("remove", [])
            }
            return bool(leaves - tried)

        ranked = sorted(
            (t for t in self.trials if t.state == "COMPLETE" and t.number <= cutoff and expandable(t)),
            key=lambda t: t.value,
            reverse=self.direction == "maximize",
        )
        if not ranked:
            return self.baseline_trial
        width = self.beam_width if self.search_strategy == "beam" else 1
        frontier = ranked[:width]
        return frontier[(number - 1) % len(frontier)]

    def _complete(self, trial, result, parent):
        """Attach evaluated metrics and joint hypothesis evidence to a trial.

        Args:
            trial (Trial): Mutable record for the current search attempt.
            result (EvaluationResult): Paired-fold metrics and optional numerical evidence.
            parent (Trial or None): Evaluated parent; None denotes the baseline.

        Returns:
            None: Mutates the trial to COMPLETE and fills deltas, timings and hypotheses.

        Notes:
            delta is relative to baseline; parent_delta is relative to the actual parent.
            Direction normalization is used only to label improvement. Permutation evidence
            and fold sign agreement do not establish individual causal feature effects.
        """
        trial.state, trial.value = "COMPLETE", result.value
        trial.metrics, trial.fold_values = result.metrics, result.fold_values
        trial.importance, trial.duration = result.importance, result.duration
        trial.transform_time, trial.training_time = result.transform_time, result.training_time
        trial.prediction_time, trial.explanation_time = result.prediction_time, result.explanation_time
        trial.cache_hit = result.cache_hit
        baseline = self.baseline_trial if trial.number else trial
        trial.delta = result.value - baseline.value
        trial.parent_delta = result.value - parent.value if parent else 0.0
        sign = 1 if self.direction == "maximize" else -1
        differences = sign * (
            np.asarray(result.fold_values) - np.asarray(parent.fold_values if parent else result.fold_values)
        )
        positive_fraction = float(np.mean(differences > 0))
        for raw in trial.features:
            name = raw["name"]
            direction = result.directions.get(name, "unknown")
            expected = raw["hypothesis"]["expected_direction"]
            importance = result.importance.get(name)
            if parent is None or name in {f["name"] for f in parent.features}:
                status, reason = "retained", "present_in_parent"
            elif sign * trial.parent_delta <= 0:
                status, reason = "rejected", "candidate_did_not_improve_parent"
            elif importance is None or importance <= 0:
                status, reason = "inconclusive", "no_positive_permutation_importance"
            elif expected != "unknown" and direction != expected:
                status = "inconclusive"
                reason = "direction_unavailable" if direction == "unknown" else "direction_conflict"
            else:
                status = "strongly_supported" if positive_fraction >= 0.8 else "supported"
                reason = "joint_gain_with_positive_importance_and_consistent_direction"
            trial.hypotheses[name] = {
                "status": status,
                "reason": reason,
                "concept": raw["hypothesis"]["concept"],
                "observed_direction": direction,
                "expected_direction": expected,
                "importance": importance,
                "positive_fold_fraction": positive_fraction,
                "stability": "high" if positive_fraction >= 0.8 else "mixed",
                "evidence_scope": "joint feature-set and model-parameter change; not individual causal attribution",
            }

    def lineage_dataframe(self):
        """Collect unique lineage nodes encountered across candidate sets.

        Returns:
            pandas.DataFrame: One row per name-sensitive feature hash, preserving first provenance.
        """
        nodes = {}
        for trial in self.trials:
            if trial.candidate_set:
                for node in FeatureSet.model_validate(trial.candidate_set).lineage():
                    nodes.setdefault(node["hash"], node)
        return pd.DataFrame(nodes.values())

    def search_summary(self, target_gain=0.005):
        """Summarize inner-search quality, resource usage and target attainment.

        Args:
            target_gain (float): Nonnegative improvement over baseline, expressed in the metric's natural
                units.

        Returns:
            dict: Efficiency metrics plus total_model_fits_including_refit.

        Notes:
            Unreached targets and unknown API costs remain None rather than zero.
        """
        from .diagnostics import search_diagnostics

        summary, _ = search_diagnostics(self.trials, self.direction, target_gain)
        summary["total_model_fits_including_refit"] = self.consumption["model_fits"]
        return summary

    def pareto_frontier(self, resource="evaluations"):
        """Return undominated completed evaluations for one resource axis.

        Args:
            resource (str): evaluations, model_fits, wall_time, tokens or api_cost for the Pareto x-axis.

        Returns:
            pandas.DataFrame: Frontier rows sorted by increasing resource cost.
        """
        from .diagnostics import pareto_frontier, search_diagnostics

        _, curve = search_diagnostics(self.trials, self.direction)
        return pareto_frontier(curve, self.direction, resource)

    def trials_dataframe(self):
        """Expose all stored trial fields for analysis.

        Returns:
            pandas.DataFrame: One row per trial, including failed and interrupted attempts.
        """
        return pd.DataFrame([trial.to_dict() for trial in self.trials])

    def feature_history(self):
        """Expand trial feature sets into expression and evidence records.

        Returns:
            pandas.DataFrame: One row per trial/feature, including retained features.
        """
        return pd.DataFrame(
            [
                {
                    "trial": trial.number,
                    "feature": feature["name"],
                    "op": feature["op"],
                    "inputs": feature["inputs"],
                    "rationale": feature["hypothesis"]["rationale"],
                    "value": trial.value,
                    "delta": trial.delta,
                    **trial.hypotheses.get(feature["name"], {}),
                }
                for trial in self.trials
                for feature in trial.features
            ]
        )

    def concept_summary(self):
        """Count recorded hypothesis statuses by semantic concept.

        Returns:
            pandas.DataFrame: concept, status and count columns; empty when no evidence exists.

        Notes:
            Counts are history entries, not independent replications or unique expressions.
        """
        history = self.feature_history()
        if history.empty or "concept" not in history:
            return pd.DataFrame(columns=["concept", "status", "count"])
        return history.groupby(["concept", "status"]).size().reset_index(name="count")

    def proposal_history(self):
        """Expose accepted proposal artifacts and sanitized attempt diagnostics.

        Returns:
            pandas.DataFrame: Candidate proposal, token/cost, context-size and failure audit rows.

        Notes:
            Raw provider text is never included. Proposals rejected inside LLMSampler may
            have empty feature lists because invalid raw responses are intentionally discarded.
        """
        return pd.DataFrame(
            [
                {
                    "trial": trial.number,
                    "stage": trial.stage,
                    "state": trial.state,
                    "valid_proposal": trial.valid_proposal,
                    "features": trial.proposal.get("features", []),
                    "removed_features": trial.proposal.get("remove", []),
                    "removal_reason": trial.context_metadata.get("removal_reason"),
                    "model_params": trial.proposal.get("model_params", {}),
                    "tokens": trial.tokens.get("input_tokens", 0) + trial.tokens.get("output_tokens", 0),
                    "api_cost": trial.api_cost,
                    "error": trial.error,
                    "screening": trial.context_metadata.get("screening"),
                    "context_columns": trial.context_metadata.get("context_columns"),
                    "estimated_prompt_tokens": trial.context_metadata.get("estimated_prompt_tokens"),
                    "retrieval_latency": trial.context_metadata.get("retrieval_latency"),
                }
                for trial in self.trials
                if trial.number > 0
            ]
        )

    def export_pipeline(self, path=None):
        """Return the refitted best pipeline and optionally write it atomically.

        Args:
            path (str, Path, or None): Optional trusted joblib destination; parent directories are
                created.

        Returns:
            sklearn.pipeline.Pipeline: The live fitted pipeline object owned by the study.

        Raises:
            ValueError: No refitted pipeline exists; run optimize with refit=True and sufficient budget.

        Notes:
            Joblib artifacts can contain learned statistics and must only be loaded from
            trusted local sources. Returning the object does not clone it.
        """
        if not hasattr(self, "pipeline_"):
            raise ValueError("Call optimize(..., refit=True) with training data before exporting")
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            joblib.dump(self.pipeline_, temporary)
            temporary.replace(path)
        return self.pipeline_

    def export_features(self, path):
        """Write the selected controlled feature proposal as readable JSON.

        Args:
            path (str or Path): Destination for the generated artifact.

        Returns:
            None: Writes/replaces the destination file.

        Notes:
            Includes best model parameters but no fitted model. The parent directory must
            already exist. Requires at least one completed trial.
        """
        from .schema import Proposal

        Path(path).write_text(
            Proposal(features=self.best_features, model_params=self.best_trial.model_params).model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

    def report(self, path=None):
        """Render a self-contained interactive HTML report of completed search evidence.

        Args:
            path (str, Path, or None): Optional output HTML path; parent directories are created.

        Returns:
            str: HTML content, also written to path when provided.

        Raises:
            ValueError: There is no completed trial to report.
            ImportError: The default Plotly dependency is missing from a broken installation.
        """
        from .reporting import render_report

        return render_report(self, path)


def create_study(**kwargs):
    """Construct a FeatureStudy using its documented configuration keywords.

    Args:
        kwargs (Any): Keyword arguments forwarded unchanged to FeatureStudy.__init__.

    Returns:
        FeatureStudy: Newly initialized or history-loaded study.
    """
    return FeatureStudy(**kwargs)


def load_study(storage, study_name="default", sampler=None):
    """Load saved configuration/history without reconstructing training data or a model.

    Args:
        storage (str or Path): Parent directory containing per-study storage directories.
        study_name (str): Single directory name identifying the study; path traversal is disallowed.
        sampler (BaseFeatureSampler or None): Proposal policy; None selects the study default.

    Returns:
        FeatureStudy: Study ready for inspection or a later compatible optimize call.

    Raises:
        ValueError: No saved study configuration is present.

    Notes:
        Provide the original nondefault sampler when resuming numerical search.
        Omitting sampler is appropriate for read-only inspection/reporting; optimize
        will validate sampler configuration before continuation.
    """
    if not study_name or Path(study_name).name != study_name or study_name in {".", ".."}:
        raise ValueError("study_name must be a single directory name")
    metadata, _ = Storage(Path(storage) / study_name).load()
    if "configuration" not in metadata:
        raise ValueError("No saved study configuration")
    config = dict(metadata["configuration"])
    for key in ("format_version", "schema", "sampler"):
        config.pop(key, None)
    return FeatureStudy(
        storage=storage, study_name=study_name, sampler=sampler, budget=metadata.get("budget"), **config
    )
