# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Deterministic semantic column retrieval, separate from generation.

Ranks compact field metadata using local lexical semantics, rules or seeded
exploration. This module neither generates features nor reads raw observations.

Created:
    2026-09-22
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .memory import SearchMemory
from .schema import DatasetSchema, FieldSchema

RETRIEVAL_VERSION = "2"


def column_concept(field: FieldSchema) -> str:
    """Choose a deterministic grouping label from field metadata.

    Args:
        field (FieldSchema): Usable column metadata; no row-level values are required.

    Returns:
        str: First semantic tag, otherwise partition, entity, or dtype.
    """
    return next(iter(field.semantic_tags), None) or field.partition or field.entity or field.dtype


@dataclass(frozen=True)
class RetrievedColumn:
    """Describe one ranked raw field without carrying row-level data.

    Attributes:
        name (str): Usable schema column name.
        score (float): Retriever-specific ranking score, not a probability.
        concept (str): Tag/partition/entity/type concept used for grouping.
        reason (str): Human-readable selection rationale.
    """

    name: str
    score: float
    concept: str
    reason: str


class ColumnSemanticIndex:
    """Index compact field semantics with local character TF-IDF.

    Attributes:
        fields (list[FieldSchema]): Usable schema fields in vector row order.
        vectorizer (TfidfVectorizer): Fitted character n-gram vocabulary.
        matrix (scipy sparse matrix): Normalized column-document vectors.

    Notes:
        No raw values, targets, full statistics, remote embeddings or API calls are used.
    """

    def __init__(self, schema: DatasetSchema):
        """Fit a lexical index over usable names, descriptions and semantic metadata.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
        """
        self.fields = list(schema.usable.values())
        documents = [
            " ".join(
                [
                    f.name.replace("_", " "),
                    f.description,
                    f.unit or "",
                    f.encoding or "",
                    f.available_at or "",
                    f.entity or "",
                    f.dtype,
                    *f.semantic_tags,
                    schema.partitions.get(f.partition, ""),
                ]
            ).lower()
            for f in self.fields
        ]
        # Character n-grams also support short names and non-Latin descriptions.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(documents)

    def scores(self, query: str) -> np.ndarray:
        """Compare task/history text against every indexed column.

        Args:
            query (str): Task and historical concept text to compare with indexed fields.

        Returns:
            numpy.ndarray: One float similarity per entry in fields, preserving index order.
        """
        return (self.matrix @ self.vectorizer.transform([query.lower()]).T).toarray().ravel()


class BaseColumnRetriever(ABC):
    """Specify the bounded column-selection interface shared by retrieval policies.

    Attributes:
        top_k (int): Maximum result count.
        concept_top_k (int): Concept cap used by semantic policies.
        seed (int): Seed for reproducible randomized policies.

    Notes:
        Implementations return ranked usable fields; ContextBuilder independently
        checks their identities and applies its own column/concept budgets.
    """

    def __init__(self, top_k: int = 40, concept_top_k: int = 5, seed: int = 42):
        """Store positive selection limits and the deterministic exploration seed.

        Args:
            top_k (int): Maximum number of returned columns; must be positive.
            concept_top_k (int): Maximum semantic concept groups retained by semantic retrieval.
            seed (int): Seed for deterministic random choices.

        Raises:
            ValueError: Column or concept limit is not positive.
        """
        if top_k < 1 or concept_top_k < 1:
            raise ValueError("Retrieval limits must be positive")
        self.top_k, self.concept_top_k, self.seed = top_k, concept_top_k, seed

    def configuration(self) -> dict:
        """Export retrieval settings used by experiment fingerprints.

        Returns:
            dict: Policy type, implementation version, limits and seed.
        """
        return {
            "type": type(self).__name__,
            "version": RETRIEVAL_VERSION,
            "top_k": self.top_k,
            "concept_top_k": self.concept_top_k,
            "seed": self.seed,
        }

    @abstractmethod
    def retrieve(self, schema: DatasetSchema, memory: SearchMemory, number: int = 0) -> list[RetrievedColumn]:
        """Select raw fields from schema and authorized inner-search memory.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            memory (SearchMemory): Bounded evidence from completed or failed inner-search trials.
            number (int): Trial index used to choose the parent or deterministic exploration seed.

        Returns:
            list[RetrievedColumn]: Ranked unique usable fields bounded by top_k.

        Notes:
            Subclasses must not read target values or outer-test outcomes. The abstract
            method defines the contract and does not produce selections itself.
        """


class RandomColumnRetriever(BaseColumnRetriever):
    """Sample schema fields without semantic ranking.

    Notes:
        Randomness is indexed by seed and trial number rather than process state.
        Configuration attributes are inherited from BaseColumnRetriever.
    """

    def retrieve(self, schema, memory, number=0):
        """Sample schema fields without semantic ranking.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            memory (SearchMemory): Bounded evidence from completed or failed inner-search trials.
            number (int): Trial index used to choose the parent or deterministic exploration seed.

        Returns:
            list[RetrievedColumn]: Ranked unique fields, no more than top_k entries.

        Notes:
            Randomness is indexed by seed and trial number rather than process state.
        """
        fields = list(schema.usable.values())
        order = np.random.default_rng(np.random.SeedSequence([self.seed, number])).permutation(len(fields))
        return [
            RetrievedColumn(fields[i].name, 0.0, column_concept(fields[i]), "seeded random exploration")
            for i in order[: self.top_k]
        ]


class SemanticColumnRetriever(BaseColumnRetriever):
    """Rank lexical semantics and expand historically successful concept neighborhoods.

    Notes:
        The schema index is reused until schema serialization changes. Similarity is
        lexical TF-IDF, not a remote embedding or synonym-understanding guarantee.
        Configuration attributes are inherited from BaseColumnRetriever.
    """

    def retrieve(self, schema, memory, number=0):
        """Rank lexical semantics and expand historically successful concept neighborhoods.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            memory (SearchMemory): Bounded evidence from completed or failed inner-search trials.
            number (int): Trial index used to choose the parent or deterministic exploration seed.

        Returns:
            list[RetrievedColumn]: Ranked unique fields, no more than top_k entries.

        Notes:
            The schema index is reused until schema serialization changes. Similarity is
            lexical TF-IDF, not a remote embedding or synonym-understanding guarantee.
        """
        identity = schema.model_dump_json()
        if getattr(self, "_identity", None) != identity:
            self._index = ColumnSemanticIndex(schema)
            self._identity = identity
        query = " ".join(
            [
                schema.objective,
                schema.target_definition or "",
                schema.prediction_point or "",
                *memory.useful_concepts,
                *(" ".join(f["inputs"]) for f in memory.best_features),
            ]
        )
        scores = self._index.scores(query)
        if schema.search_guidance:
            scores += 2 * self._index.scores(schema.search_guidance)
        fields = self._index.fields
        useful = set(memory.useful_concepts)
        # Expand the neighborhoods of actually successful input fields.
        known = schema.usable
        for interaction in memory.strong_interactions:
            useful.update(column_concept(known[name]) for name in interaction["inputs"] if name in known)
        for i, field in enumerate(fields):
            if column_concept(field) in useful:
                scores[i] += 0.25
        order = sorted(range(len(fields)), key=lambda i: (-scores[i], fields[i].name))
        concepts = list(dict.fromkeys(column_concept(fields[i]) for i in order))[: self.concept_top_k]
        return [
            RetrievedColumn(
                fields[i].name,
                float(scores[i]),
                column_concept(fields[i]),
                "validated concept neighborhood"
                if column_concept(fields[i]) in useful
                else "task lexical similarity",
            )
            for i in order
            if column_concept(fields[i]) in concepts
        ][: self.top_k]


class RuleBasedColumnRetriever(BaseColumnRetriever):
    """Rank fields using deterministic name, tag and unit matches.

    Notes:
        Ties are resolved by field name; unmatched fields remain available for exploration.
        Configuration attributes are inherited from BaseColumnRetriever.
    """

    def retrieve(self, schema, memory, number=0):
        """Rank fields using deterministic name, tag and unit matches.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            memory (SearchMemory): Bounded evidence from completed or failed inner-search trials.
            number (int): Trial index used to choose the parent or deterministic exploration seed.

        Returns:
            list[RetrievedColumn]: Ranked unique fields, no more than top_k entries.

        Notes:
            Ties are resolved by field name; unmatched fields remain available for exploration.
        """
        query = " ".join([schema.objective, *memory.useful_concepts]).lower()
        fields = list(schema.usable.values())
        scores = {
            f.name: sum(token.lower() in query for token in [f.name, *f.semantic_tags, f.unit or "\0"])
            for f in fields
        }
        ordered = sorted(fields, key=lambda f: (-scores[f.name], f.name))
        return [
            RetrievedColumn(f.name, float(scores[f.name]), column_concept(f), "name/tag/unit rule")
            for f in ordered[: self.top_k]
        ]


class HybridColumnRetriever(BaseColumnRetriever):
    """Combine semantic exploitation with seeded random exploration.

    Notes:
        Approximately half the slots use semantic ranking; remaining slots contain
        unique randomly selected fields, subject to the final ContextBuilder budget.
        Configuration attributes are inherited from BaseColumnRetriever.
    """

    def retrieve(self, schema, memory, number=0):
        # Retain the index across rounds, while reserving half the slots for exploration.
        """Combine semantic exploitation with seeded random exploration.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            memory (SearchMemory): Bounded evidence from completed or failed inner-search trials.
            number (int): Trial index used to choose the parent or deterministic exploration seed.

        Returns:
            list[RetrievedColumn]: Ranked unique fields, no more than top_k entries.

        Notes:
            Approximately half the slots use semantic ranking; remaining slots contain
            unique randomly selected fields, subject to the final ContextBuilder budget.
        """
        if not hasattr(self, "_semantic"):
            self._semantic = SemanticColumnRetriever(self.top_k, self.concept_top_k, self.seed)
        ranked = self._semantic.retrieve(schema, memory, number)[: max(1, self.top_k // 2)]
        seen = {column.name for column in ranked}
        random = RandomColumnRetriever(len(schema.usable), self.concept_top_k, self.seed)
        return (
            ranked + [column for column in random.retrieve(schema, memory, number) if column.name not in seen]
        )[: self.top_k]
