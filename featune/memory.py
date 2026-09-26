# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Bounded, evidence-only search memory for retrieval and prompts.

Compresses existing trial evidence deterministically. Summaries are bounded
heuristics, not generated claims of causality or statistical significance.

Created:
    2026-09-22
"""

from collections import Counter

from pydantic import Field

from .schema import Contract

MEMORY_VERSION = "1"


class SearchMemory(Contract):
    """Store bounded summaries grounded in actual inner-search evidence.

    Attributes:
        version (str): Compression format version for fingerprinting.
        useful_concepts (list[str]): Concepts from proposals with positive direction-adjusted parent
            delta.
        rejected_concepts (list[str]): Concepts from completed non-improving proposals.
        best_features (list[dict]): Ranked proposed expressions with inner-trial evidence.
        failed_operator_patterns (list[str]): Frequent failed or non-improving operators.
        strong_interactions (list[dict]): Recent two-input proposals in improving joint feature sets.
        weak_interactions (list[dict]): Recent two-input proposals in non-improving sets.
        invalid_patterns (dict[str, int]): Sanitized validation categories or failure stages and
            counts.
        recent_frontier (list[dict]): Bounded recent trial numbers, states, values and parent deltas.

    Notes:
        Strong/weak are heuristic joint-set labels, not individual significance or
        causal evidence. ContextBuilder enforces prompt-size limits after compression.
    """

    version: str = MEMORY_VERSION
    useful_concepts: list[str] = Field(default_factory=list)
    rejected_concepts: list[str] = Field(default_factory=list)
    best_features: list[dict] = Field(default_factory=list)
    failed_operator_patterns: list[str] = Field(default_factory=list)
    strong_interactions: list[dict] = Field(default_factory=list)
    weak_interactions: list[dict] = Field(default_factory=list)
    invalid_patterns: dict[str, int] = Field(default_factory=dict)
    recent_frontier: list[dict] = Field(default_factory=list)


class SearchMemoryCompressor:
    """Reduce recorded trials to deterministic, bounded prompt/retrieval summaries.

    Notes:
        No model-generated narrative or outer-test results participate in compression.
    """

    def compress(
        self,
        history: list[dict],
        direction: str,
        limit: int = 10,
        max_concepts: int = 6,
        max_failed_patterns: int = 10,
    ) -> SearchMemory:
        """Summarize real successes, failures and the recent exploration frontier.

        Args:
            history (list[dict]): Recorded inner-search trials; do not supply outer-test outcomes.
            direction (str): "maximize" for rewards or "minimize" for losses.
            limit (int): Maximum retained best-feature, interaction and recent-history entries per
                category.
            max_concepts (int): Maximum retained useful/rejected concept labels per category.
            max_failed_patterns (int): Maximum retained failed-operator and invalid-pattern entries per
                category.

        Returns:
            SearchMemory: Bounded concepts, expressions, interactions and sanitized failure counts.

        Notes:
            Only new proposal features inherit a trial's parent delta. Retained features
            and duplicate evaluations are not counted again as new positive evidence.
            Limits must be nonnegative; callers normally use validated ContextBudget values.
            Compression scans history and sorts completed trials; the returned memory stays bounded.
        """
        sign = 1 if direction == "maximize" else -1
        complete = [t for t in history if t.get("state") == "COMPLETE" and t.get("number", 0) > 0]
        ranked = sorted(complete, key=lambda t: (-sign * t["value"], t["number"]))
        useful, rejected, failed, invalid = Counter(), Counter(), Counter(), Counter()
        strong, weak, best = [], [], []
        for trial in history:
            patterns = trial.get("context_metadata", {}).get("invalid_patterns", {})
            invalid.update(patterns)
            if trial.get("state") == "FAIL" and not patterns:
                invalid[trial.get("stage", "proposal")] += 1
            # Only newly proposed features inherit this trial's incremental evidence.
            features = trial.get("proposal", {}).get("features", [])
            delta = trial.get("parent_delta")
            if trial.get("state") != "COMPLETE" or delta is None:
                if trial.get("state") == "FAIL":
                    failed.update(f["op"] for f in features)
                continue
            improved = sign * delta > 0
            for feature in features:
                (useful if improved else rejected)[feature["hypothesis"]["concept"]] += 1
                if not improved:
                    failed[feature["op"]] += 1
                if len(feature["inputs"]) == 2:
                    (strong if improved else weak).append(
                        {
                            "inputs": feature["inputs"],
                            "op": feature["op"],
                            "trial": trial["number"],
                            "parent_delta": delta,
                        }
                    )
        for trial in ranked:
            for feature in trial.get("proposal", {}).get("features", []):
                item = {key: feature[key] for key in ("name", "op", "inputs")}
                item.update(concept=feature["hypothesis"]["concept"], trial=trial["number"])
                item.update(
                    params=feature.get("params", {}),
                    importance=trial.get("importance", {}).get(feature["name"]),
                    parent_delta=trial.get("parent_delta"),
                )
                if item not in best:
                    best.append(item)
            if len(best) >= limit:
                break
        frontier = (
            [
                {key: t.get(key) for key in ("number", "state", "value", "parent_delta")}
                for t in history[-limit:]
            ]
            if limit
            else []
        )
        return SearchMemory(
            useful_concepts=[name for name, _ in useful.most_common(max_concepts)],
            rejected_concepts=[name for name, _ in rejected.most_common(max_concepts)],
            best_features=best[:limit],
            failed_operator_patterns=[name for name, _ in failed.most_common(max_failed_patterns)],
            strong_interactions=strong[-limit:] if limit else [],
            weak_interactions=weak[-limit:] if limit else [],
            invalid_patterns=dict(invalid.most_common(max_failed_patterns)),
            recent_frontier=frontier,
        )
