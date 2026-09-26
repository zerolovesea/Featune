# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Bounded prompt construction and selective training statistics.

Allocates prompt information after bounded retrieval. Only selected target-free
statistics and compressed inner-search evidence enter generated prompts.

Created:
    2026-09-22
"""

import json
import time
from dataclasses import asdict, dataclass

from pydantic import Field, model_validator

from .ir import content_hash
from .llm import LLMError
from .memory import MEMORY_VERSION, SearchMemory, SearchMemoryCompressor
from .retrieval import BaseColumnRetriever, RetrievedColumn, SemanticColumnRetriever, column_concept
from .schema import OPS, Contract

CONTEXT_VERSION = "2"


def serialize(value) -> str:
    """Serialize compact prompt data without accepting non-finite values.

    Args:
        value (JSON-serializable object): Selected, policy-compliant context payload.

    Returns:
        str: Compact Unicode JSON preserving insertion order.

    Raises:
        ValueError: A payload number is non-finite.
        TypeError: A payload object is not JSON serializable.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def estimate_tokens(text: str) -> int:
    """Compute a conservative text-token reserve from UTF-8 byte length.

    Args:
        text (str): Unicode text whose UTF-8 representation is measured.

    Returns:
        int: Prompt-text byte count; not provider tokenization or billing.
    """
    return len(text.encode("utf-8"))


class ContextBudget(Contract):
    """Define independent limits for a single LLM prompt.

    Attributes:
        max_columns (int): Positive hard cap on selected raw fields.
        max_detailed_stats_columns (int): Maximum fields whose detailed statistics may be requested.
        max_history_trials (int): Per-category limit for history-derived feature/interaction/frontier
            entries.
        max_failed_patterns (int): Per-category limit for failed operators and invalid-pattern
            summaries.
        max_tokens (int): Hard conservative UTF-8 byte reserve for prompt text, not billed usage.
        max_concepts (int): Maximum selected and summarized concept labels per category.
        max_lineage_items (int): Maximum additional lineage detail entries; required parent inputs are
            separate.
        token_allocation (dict[str, float]): Positive schema/memory/statistics/instructions fractions
            summing to one.

    Notes:
        Information allocations are soft; required task/DSL/parent constraints may borrow
        unused capacity. The total prompt token reserve remains a hard limit.
    """

    max_columns: int = Field(default=40, ge=1)
    max_detailed_stats_columns: int = Field(default=20, ge=0)
    max_history_trials: int = Field(default=10, ge=0)
    max_failed_patterns: int = Field(default=10, ge=0)
    max_tokens: int = Field(default=12000, ge=1)
    max_concepts: int = Field(default=6, ge=1)
    max_lineage_items: int = Field(default=20, ge=0)
    token_allocation: dict[str, float] = Field(
        default_factory=lambda: {"schema": 0.4, "memory": 0.25, "statistics": 0.2, "instructions": 0.15}
    )

    @model_validator(mode="after")
    def validate_allocation(self):
        """Validate names and normalization of soft token-allocation fractions.

        Returns:
            ContextBudget: This validated budget.

        Raises:
            ValueError: Keys are missing/extra, fractions are nonpositive, or their sum differs from one.
        """
        if (
            set(self.token_allocation) != {"schema", "memory", "statistics", "instructions"}
            or any(value <= 0 for value in self.token_allocation.values())
            or abs(sum(self.token_allocation.values()) - 1) > 1e-6
        ):
            raise ValueError("Token allocation must contain four positive fractions summing to one")
        return self


@dataclass
class BuiltContext:
    """Return a bounded prompt together with selected fields and audit metadata.

    Attributes:
        prompt (str): Ready-to-send prompt containing only authorized summaries.
        columns (list[str]): Final selected raw-column names after trimming.
        metadata (dict): Selection scores/reasons, timings, version, prompt hash and token reserve.
    """

    prompt: str
    columns: list[str]
    metadata: dict


class ContextBuilder:
    """Separate field selection from prompt information allocation and trimming.

    Attributes:
        budget (ContextBudget): Validated per-request information limits.
        retriever (BaseColumnRetriever): Field selection policy; may retain a schema index.
        retrieval_threshold (int): Automatic retrieval threshold, subordinate to the column hard
            limit.
        use_memory (bool): Include compressed inner-trial evidence when True.

    Notes:
        The builder has no row-data or outer-test argument. Optional statistics are
        requested lazily from SearchContext.statistics_provider for selected fields.
    """

    def __init__(
        self,
        budget: ContextBudget | dict | None = None,
        retriever: BaseColumnRetriever | None = None,
        retrieval_threshold: int = 200,
        use_memory: bool = True,
    ):
        """Configure bounded context construction independently of the LLM client.

        Args:
            budget (ContextBudget, dict, or None): Per-prompt limits; None uses defaults.
            retriever (BaseColumnRetriever or None): Policy selecting fields; None uses local semantic
                retrieval.
            retrieval_threshold (int): Column count enabling retrieval; the column budget can enable it
                sooner.
            use_memory (bool): Whether real trial history influences retrieval and prompt summaries.

        Raises:
            ValueError: The retrieval threshold or context budget is invalid.
        """
        if retrieval_threshold < 1:
            raise ValueError("retrieval_threshold must be positive")
        self.budget = ContextBudget.model_validate(budget or {})
        self.retriever = retriever or SemanticColumnRetriever(
            top_k=self.budget.max_columns, concept_top_k=self.budget.max_concepts
        )
        self.retrieval_threshold = retrieval_threshold
        self.use_memory = use_memory

    def configuration(self) -> dict:
        """Export deterministic context settings for compatibility fingerprints.

        Returns:
            dict: Budget, retriever configuration, thresholds and component versions; no runtime timings.
        """
        return {
            "version": CONTEXT_VERSION,
            "memory_version": MEMORY_VERSION,
            "budget": self.budget.model_dump(),
            "retriever": self.retriever.configuration(),
            "retrieval_threshold": self.retrieval_threshold,
            "use_memory": self.use_memory,
        }

    def build(self, context, instruction: str, max_chars: int, max_new_features: int) -> BuiltContext:
        """Retrieve fields, enrich selected statistics and deterministically trim a prompt.

        Args:
            context (SearchContext): Schema, current parent, inner-trial evidence and remaining budgets.
            instruction (str): Mandatory task-independent DSL/proposal instructions, retained during
                trimming.
            max_chars (int): Hard character ceiling after any caller-reserved repair suffixes.
            max_new_features (int): Requested proposal size, also bounded by remaining feature capacity.

        Returns:
            BuiltContext: Final prompt, allowed raw-field names and persistable audit metadata.

        Raises:
            ValueError: A retriever returns duplicate, excluded or unknown columns.
            LLMError: No usable column remains or mandatory constraints exceed the prompt budget.

        Notes:
            May populate the retriever index and call statistics_provider for selected fields.
            Only whitelisted summary keys are serialized. Statistics are dropped before
            low-priority memory/lineage/columns; required task and DSL rules are never truncated.
            The caller must reserve repair suffixes before invoking this method.
        """
        start = time.monotonic()
        budget = self.budget
        memory = (
            SearchMemoryCompressor().compress(
                context.history,
                context.direction,
                budget.max_history_trials,
                budget.max_concepts,
                budget.max_failed_patterns,
            )
            if self.use_memory
            else SearchMemory()
        )
        # The hard column cap still applies when the retrieval threshold is configured higher.
        retrieving = (
            len(context.schema.usable) >= self.retrieval_threshold
            or len(context.schema.usable) > budget.max_columns
        )
        if retrieving:
            selected = self.retriever.retrieve(context.schema, memory, context.number)[: budget.max_columns]
        else:
            selected = [
                RetrievedColumn(name, 0.0, column_concept(field), "compact schema below threshold")
                for name, field in context.schema.usable.items()
            ]
        # Bound custom retrievers as well; excluded or invented names are never accepted.
        if len({item.name for item in selected}) != len(selected) or any(
            item.name not in context.schema.usable for item in selected
        ):
            raise ValueError("Retriever returned duplicate, unknown or excluded fields")
        selected_concepts = list(dict.fromkeys(item.concept for item in selected))[: budget.max_concepts]
        selected = [item for item in selected if item.concept in selected_concepts]
        if not selected:
            raise LLMError("Retriever selected no usable columns")
        retrieval_latency = time.monotonic() - start
        fields = context.schema.usable
        compact = [
            {
                "name": item.name,
                "dtype": fields[item.name].dtype,
                "description": fields[item.name].description[:240],
                "unit": fields[item.name].unit,
                "encoding": fields[item.name].encoding,
                "available_at": fields[item.name].available_at,
                "concept": item.concept,
            }
            for item in selected
        ]
        schema_limit = int(budget.max_tokens * budget.token_allocation["schema"])
        while len(compact) > 1 and estimate_tokens(serialize(compact)) > schema_limit:
            compact.pop()
        names = [column["name"] for column in compact]
        stats_names = names[: budget.max_detailed_stats_columns]
        statistics = (
            context.statistics_provider(stats_names)
            if context.statistics_provider and stats_names
            else {
                name: context.column_statistics[name]
                for name in stats_names
                if name in context.column_statistics
            }
        )
        # Whitelist summary keys; never serialize arbitrary caller metadata or row samples.
        allowed_stats = {
            "dtype",
            "missing_rate",
            "cardinality",
            "min",
            "max",
            "mean",
            "std",
            "quantiles",
            "top_category_frequencies",
            "skewness",
            "zero_rate",
            "inf_rate",
        }
        statistics = {
            name: {key: value for key, value in summary.items() if key in allowed_stats}
            for name, summary in statistics.items()
            if name in stats_names
        }
        while (
            statistics
            and estimate_tokens(serialize(statistics))
            > budget.max_tokens * budget.token_allocation["statistics"]
        ):
            statistics.pop(next(reversed(statistics)))
        memory_data = memory.model_dump()
        memory_limit = int(budget.max_tokens * budget.token_allocation["memory"])
        self._trim_memory(memory_data, memory_limit)
        data = {
            "task": context.schema.objective,
            "target_definition": context.schema.target_definition,
            "prediction_point": context.schema.prediction_point,
            "estimator": context.estimator,
            "columns": compact,
            "concepts": selected_concepts,
            "metric": context.metric,
            "direction": context.direction,
            "parent_features": [
                {"name": f.name, "op": f.op, "inputs": f.inputs, "params": f.params}
                for f in context.parent_features
            ],
            "parent_model_params": context.parent_params,
            "approved_parameter_choices": context.param_space,
            "max_new_features": min(context.remaining_features, max_new_features),
            "history": memory_data,
            "operators": OPS,
            "column_statistics": statistics,
            "lineage": [
                {key: item.get(key) for key in ("name", "op", "inputs", "generation", "content_hash")}
                for item in context.current_feature_set.get("features", [])[-budget.max_lineage_items :]
            ]
            if budget.max_lineage_items
            else [],
            "current_feature_set_hash": context.current_feature_set.get("content_hash"),
            "remaining_budget": context.remaining_budget,
        }
        while True:
            prompt = instruction + "\nContext:\n" + serialize(data)
            if len(prompt) <= max_chars and estimate_tokens(prompt) <= budget.max_tokens:
                break
            if data["column_statistics"]:
                data["column_statistics"].pop(next(reversed(data["column_statistics"])))
            elif self._drop_memory_item(data["history"]):
                continue
            elif data["lineage"]:
                data["lineage"].pop(0)
            elif len(data["columns"]) > 1:
                data["columns"].pop()
            else:
                raise LLMError("Task, DSL and parent exceed max_context_chars or ContextBudget.max_tokens")
        names = [column["name"] for column in data["columns"]]
        metadata = {
            "version": CONTEXT_VERSION,
            "memory_version": MEMORY_VERSION,
            "retrieval_enabled": retrieving,
            "columns": names,
            "context_columns": len(names),
            "detailed_statistics_columns": list(data["column_statistics"]),
            "estimated_prompt_tokens": estimate_tokens(prompt),
            "prompt_hash": content_hash(prompt),
            "retrieval_latency": retrieval_latency,
            "build_time": time.monotonic() - start,
            "selection": [asdict(item) for item in selected if item.name in names],
        }
        return BuiltContext(prompt, names, metadata)

    @staticmethod
    def _drop_memory_item(memory: dict) -> bool:
        """Remove one low-priority memory entry in place.

        Args:
            memory (dict): Mutable SearchMemory payload containing every expected collection.

        Returns:
            bool: True if an entry was removed; False when all removable collections are empty.

        Notes:
            Recent frontier trimming removes the oldest entry; ranked feature/concept lists
            lose their lowest-priority tail item.
        """
        for key in (
            "invalid_patterns",
            "failed_operator_patterns",
            "weak_interactions",
            "rejected_concepts",
            "recent_frontier",
            "strong_interactions",
            "best_features",
            "useful_concepts",
        ):
            if memory[key]:
                if isinstance(memory[key], dict):
                    memory[key].pop(next(reversed(memory[key])))
                else:
                    memory[key].pop(0 if key == "recent_frontier" else -1)
                return True
        return False

    @classmethod
    def _trim_memory(cls, memory: dict, max_tokens: int):
        """Shrink memory in place to its soft token allocation when possible.

        Args:
            memory (dict): Mutable serialized SearchMemory payload.
            max_tokens (int): Soft UTF-8 byte allocation for memory JSON.

        Returns:
            None: Mutates memory until within the allocation or only empty fields remain.
        """
        while estimate_tokens(serialize(memory)) > max_tokens and cls._drop_memory_item(memory):
            pass
