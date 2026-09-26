# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Canonical feature IR, feature sets and lineage DAGs.

Normalizes controlled expressions into auditable dependency graphs. Name-sensitive
    lineage identity is separate from alias-independent expression identity.

Created:
    2026-09-21
"""

from __future__ import annotations

import hashlib
import json
from graphlib import CycleError, TopologicalSorter
from typing import Literal

from pydantic import Field, model_validator

from .schema import Contract, DatasetSchema, FeatureSpec, validate_features

DSL_VERSION = "1"
COMPILER_VERSION = "3"
PROMPT_VERSION = "4"


def content_hash(value) -> str:
    """Hash a JSON-compatible value using canonical UTF-8 serialization.

    Args:
        value (JSON-serializable object): Identity payload; dictionary keys are sorted.

    Returns:
        str: Hexadecimal SHA-256 digest.

    Raises:
        ValueError: A numeric value is non-finite.
        TypeError: A value cannot be serialized as JSON.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


class FeatureIR(FeatureSpec):
    """Attach validated execution identity and lineage to a FeatureSpec.

    Attributes:
        output_dtype (str): numeric or categorical, inferred from the operator.
        stateful (bool): Whether the operator learns training statistics.
        lineage_parents (dict[str, str]): Input names mapped to raw-field or parent-feature hashes.
        generation (int): One plus the deepest derived parent generation.
        source_trial (int or None): Trial that first introduced this node.
        content_hash (str): Name-sensitive DSL/version/operator/input/parameter digest.
        version (str): Supported DSL version.

    Notes:
        Inherits name, op, inputs, params and hypothesis from FeatureSpec.
        Hash consistency is validated when an IR object is constructed or loaded.
    """

    output_dtype: Literal["numeric", "categorical"]
    stateful: bool
    lineage_parents: dict[str, str]
    generation: int = Field(ge=1)
    source_trial: int | None = Field(default=None, ge=0)
    content_hash: str
    version: str = DSL_VERSION

    @model_validator(mode="after")
    def validate_ir(self):
        """Verify metadata, version, lineage keys and the content digest.

        Returns:
            FeatureIR: This validated node.

        Raises:
            ValueError: Operator metadata, parent keys, version or digest are inconsistent.
        """
        expected_dtype = "categorical" if self.op == "cross" else "numeric"
        expected_stateful = self.op in {"frequency", "group_mean", "group_std"}
        if self.output_dtype != expected_dtype or self.stateful != expected_stateful:
            raise ValueError("IR output dtype/stateful metadata does not match its operator")
        if self.version != DSL_VERSION or set(self.lineage_parents) != set(self.inputs):
            raise ValueError("Invalid IR version or lineage parents")
        if self.content_hash != self.calculate_hash(self.spec, self.lineage_parents):
            raise ValueError("IR content hash mismatch")
        return self

    @property
    def spec(self) -> FeatureSpec:
        """Project execution metadata back to the controlled proposal representation.

        Returns:
            FeatureSpec: A validated spec retaining the expression and hypothesis.
        """
        return FeatureSpec(**{name: getattr(self, name) for name in FeatureSpec.model_fields})

    @staticmethod
    def calculate_hash(spec, parents):
        """Build a node digest from its display name and parent identities.

        Args:
            spec (FeatureSpec): Validated expression whose hypothesis/provenance are not hashed.
            parents (dict[str, str]): Every input name mapped to its identity digest.

        Returns:
            str: Canonical SHA-256 digest.

        Notes:
            Commutative input hashes are sorted; rationale and source trial do not affect identity.
        """
        inputs = [parents[name] for name in spec.inputs]
        if spec.op in {"add", "multiply", "cross"}:
            inputs.sort()
        return content_hash(
            {
                "version": DSL_VERSION,
                "name": spec.name,
                "operator": spec.op,
                "inputs": inputs,
                "params": spec.params,
            }
        )


class FeatureSet(Contract):
    """Represent a validated feature graph with order-independent collection identity.

    Attributes:
        features (tuple[FeatureIR, ...]): Nodes with unique names and consistent acyclic lineage.
        content_hash (str): Hash of sorted node content hashes and DSL version.
        version (str): Supported DSL version.

    Notes:
        Use from_features to normalize execution order. content_hash preserves aliases
        for lineage; expression_hash ignores derived aliases for comparison only.
    """

    features: tuple[FeatureIR, ...] = ()
    content_hash: str
    version: str = DSL_VERSION

    @model_validator(mode="after")
    def validate_set(self):
        """Check collection identity, unique names and graph consistency.

        Returns:
            FeatureSet: This validated graph.

        Raises:
            ValueError: Version/hash, names, cycles, parent hashes or generations are invalid.
        """
        if self.version != DSL_VERSION or self.content_hash != content_hash(
            [DSL_VERSION, sorted(feature.content_hash for feature in self.features)]
        ):
            raise ValueError("FeatureSet hash/version mismatch")
        names = [feature.name for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate FeatureSet names")
        graph = {f.name: set(f.inputs) & set(names) for f in self.features}
        try:
            tuple(TopologicalSorter(graph).static_order())
        except CycleError:
            raise ValueError("Cyclic feature lineage") from None
        by_name = {feature.name: feature for feature in self.features}
        for feature in self.features:
            if any(
                feature.lineage_parents[name] != by_name[name].content_hash for name in graph[feature.name]
            ):
                raise ValueError("Inconsistent lineage parent hash")
            expected_generation = 1 + max(
                (by_name[name].generation if name in by_name else 0 for name in feature.inputs), default=0
            )
            if feature.generation != expected_generation:
                raise ValueError("Inconsistent lineage generation")
        return self

    @classmethod
    def from_features(cls, schema: DatasetSchema, features, source_trial=None, origins=None):
        """Normalize controlled specs into a validated, deterministically ordered IR graph.

        Args:
            schema (DatasetSchema): Validated field types, semantics, exclusions and task objective.
            features (FeatureSet or sequence[FeatureSpec]): Controlled derived features, excluding raw
                columns.
            source_trial (int or None): Provenance assigned to newly created nodes.
            origins (FeatureSet or None): Existing parent nodes whose provenance should be retained.

        Returns:
            FeatureSet: Nodes ordered by generation then recursive expression identity.

        Raises:
            ValueError: Names, references, types, cycles or feature limits violate the DSL.

        Notes:
            Unlike raw sampler validation, this factory topologically sorts supplied specs.
            Raw-field hashes include name and dtype; full schema compatibility is checked by
            the evaluator/fingerprint rather than by each individual node hash.
        """
        raw = (
            features.to_specs()
            if isinstance(features, cls)
            else [
                feature.spec if isinstance(feature, FeatureIR) else FeatureSpec.model_validate(feature)
                for feature in features
            ]
        )
        names = {feature.name for feature in raw}
        if len(names) != len(raw):
            raise ValueError("Duplicate FeatureSet names")
        by_name = {feature.name: feature for feature in raw}
        graph = {name: sorted(set(by_name[name].inputs) & names) for name in sorted(names)}
        try:
            order = list(TopologicalSorter(graph).static_order())
        except CycleError:
            raise ValueError("Cyclic feature lineage") from None
        ordered = [by_name[name] for name in order]
        validate_features(schema, ordered)
        hashes = {name: content_hash(["field", name, field.dtype]) for name, field in schema.usable.items()}
        generations = dict.fromkeys(hashes, 0)
        previous = {f.content_hash: f for f in origins.features} if origins else {}
        nodes = []
        for feature in ordered:
            parents = {name: hashes[name] for name in feature.inputs}
            digest = FeatureIR.calculate_hash(feature, parents)
            generation = max(generations[name] for name in feature.inputs) + 1
            node = previous.get(digest) or FeatureIR(
                **feature.model_dump(),
                output_dtype="categorical" if feature.op == "cross" else "numeric",
                stateful=feature.op in {"frequency", "group_mean", "group_std"},
                lineage_parents=parents,
                generation=generation,
                source_trial=source_trial,
                content_hash=digest,
            )
            nodes.append(node)
            hashes[feature.name], generations[feature.name] = digest, generation
        result = cls(
            features=tuple(nodes),
            content_hash=content_hash([DSL_VERSION, sorted(f.content_hash for f in nodes)]),
        )
        expressions = result.expression_hashes()
        # Estimators see a stable column order even when proposal display names change.
        return result.model_copy(
            update={
                "features": tuple(sorted(nodes, key=lambda node: (node.generation, expressions[node.name])))
            }
        )

    def to_specs(self) -> list[FeatureSpec]:
        """Remove execution metadata from the graph nodes.

        Returns:
            list[FeatureSpec]: Validated specifications in the stored execution order.
        """
        return [feature.spec for feature in self.features]

    def expression_hashes(self) -> dict[str, str]:
        """Resolve recursive computation identities independently of display aliases.

        Returns:
            dict[str, str]: Derived output name to recursive expression digest.

        Notes:
            Raw-column identity remains name/type sensitive. Parents are resolved before
            children using generation order; commutative operator inputs are normalized.
        """
        hashes = {}
        for feature in sorted(self.features, key=lambda item: (item.generation, item.name)):
            inputs = [hashes.get(name, feature.lineage_parents[name]) for name in feature.inputs]
            if feature.op in {"add", "multiply", "cross"}:
                inputs.sort()
            hashes[feature.name] = content_hash(["expression-v1", feature.op, inputs, feature.params])
        return hashes

    @property
    def expression_hash(self) -> str:
        """Identify the collection by recursive expressions rather than display names.

        Returns:
            str: Hash of sorted expression identities, retaining expression multiplicity.
        """
        return content_hash(["expression-set-v1", sorted(self.expression_hashes().values())])

    def lineage(self) -> list[dict]:
        """Export auditable node and parent relationships.

        Returns:
            list[dict]: Feature names, operators, parameters, concepts, provenance and hashes.
        """
        return [
            {
                "feature": feature.name,
                "parents": feature.inputs,
                "operator": feature.op,
                "params": feature.params,
                "trial_id": feature.source_trial,
                "generation": feature.generation,
                "concept": feature.hypothesis.concept,
                "hash": feature.content_hash,
                "parent_hashes": feature.lineage_parents,
            }
            for feature in self.features
        ]
