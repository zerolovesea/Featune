# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Version-namespaced atomic memoization for trusted local studies.

Stores trusted local joblib artifacts. Namespace compatibility is supplied by
the caller; this module does not validate external artifact provenance.

Created:
    2026-09-21
"""

import uuid
from pathlib import Path

import joblib

from .ir import content_hash


class ArtifactCache:
    """Memoize trusted artifacts in memory or a versioned local directory.

    Attributes:
        namespace (str): Experiment namespace separating incompatible artifacts.
        directory (Path or None): Namespace directory, or None for memory-only caching.
        memory (dict): In-process entries keyed by layer and key digest.
        hits (int): Successful get operations since construction.
        misses (int): Unsuccessful get operations since construction.

    Notes:
        Disk artifacts use joblib and must never be loaded from untrusted sources.
        Heavy memory layers retain at most eight insertion-ordered entries each.
    """

    layers = {"ir", "feature_set", "fold_transform", "matrix", "estimator", "evaluation"}

    def __init__(self, directory=None, namespace="memory"):
        """Initialize an empty cache and resource counters.

        Args:
            directory (str, Path, or None): Cache root; None selects in-memory storage.
            namespace (str): Experiment identity separating incompatible cache entries.
        """
        self.namespace = namespace
        self.directory = Path(directory) / namespace if directory else None
        self.memory = {}
        self.hits, self.misses = 0, 0

    def _path(self, layer, key):
        """Resolve a layer/key to its namespaced artifact path.

        Args:
            layer (str): One of ir, feature_set, fold_transform, matrix, estimator or evaluation.
            key (JSON-serializable object): Deterministic artifact identity within the namespace.

        Returns:
            Path or None: Disk destination, or None in memory mode.

        Raises:
            ValueError: layer is not a registered cache layer.
        """
        if layer not in self.layers:
            raise ValueError("Unknown cache layer")
        return self.directory / layer / (content_hash(key) + ".joblib") if self.directory else None

    def get(self, layer, key):
        """Load an artifact and increment hit or miss accounting.

        Args:
            layer (str): One of ir, feature_set, fold_transform, matrix, estimator or evaluation.
            key (JSON-serializable object): Deterministic artifact identity within the namespace.

        Returns:
            Any or None: Cached object, or None on a miss.

        Notes:
            Does not copy in-memory objects. Corrupt disk artifacts and I/O errors propagate
            instead of silently returning an unrelated or incomplete result.
        """
        path = self._path(layer, key)
        if path is not None and path.exists():
            self.hits += 1
            return joblib.load(path)
        memory_key = (layer, content_hash(key))
        if path is None and memory_key in self.memory:
            self.hits += 1
            return self.memory[memory_key]
        self.misses += 1
        return None

    def put(self, layer, key, value):
        """Store an artifact atomically on disk or with bounded in-memory retention.

        Args:
            layer (str): One of ir, feature_set, fold_transform, matrix, estimator or evaluation.
            key (JSON-serializable object): Deterministic artifact identity within the namespace.
            value (Any): Trusted joblib-serializable artifact; callers treat retrieved objects as shared.

        Returns:
            None: Mutates cache storage.

        Notes:
            Disk writes use a unique sibling temporary file followed by replacement, so
            readers never observe a partially serialized destination.
        """
        path = self._path(layer, key)
        if path is None:
            self.memory[layer, content_hash(key)] = value
            if layer in {"matrix", "fold_transform", "estimator"}:
                # ponytail: retain eight heavy entries per layer; use disk storage for larger searches.
                entries = [item for item in self.memory if item[0] == layer]
                if len(entries) > 8:
                    del self.memory[entries[0]]
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            joblib.dump(value, temporary)
            temporary.replace(path)
