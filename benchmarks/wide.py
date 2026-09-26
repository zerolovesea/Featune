# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Reproducible wide-context ablations; offline by default.

Defaults to offline retrieval smoke; --llm explicitly enables paid proposal
evaluation. Synthetic recall proxies do not establish real-world superiority.

Created:
    2026-09-22
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from featune import (
    ContextBudget,
    ContextBuilder,
    CVEvaluator,
    DatasetSchema,
    FieldSchema,
    LLMSampler,
    RandomColumnRetriever,
    SearchBudget,
    SemanticColumnRetriever,
    create_study,
)
from featune.samplers import PROPOSAL_INSTRUCTION, SearchContext

STRATEGIES = ["full", "random", "semantic", "semantic_grouping", "semantic_memory"]


def builder_for(strategy: str, size: int, seed: int) -> ContextBuilder:
    """Construct a context policy for a named wide-schema ablation.

    Args:
        strategy (str): full, random, semantic, semantic_grouping or semantic_memory.
        size (int): Number of columns in the synthetic wide schema.
        seed (int): Seed for deterministic random choices.

    Returns:
        ContextBuilder: Full-context or bounded retrieval/grouping/memory configuration.

    Notes:
        Full-context explicitly enlarges limits and is safe only when the chosen model
        window permits it. Budgeted policies retain at most forty selected raw fields.
    """
    full = strategy == "full"
    concepts = 5 if strategy in {"semantic_grouping", "semantic_memory"} else size
    budget = ContextBudget(
        max_columns=size if full else 40, max_tokens=500_000 if full else 12000, max_concepts=concepts
    )
    kind = RandomColumnRetriever if strategy == "random" else SemanticColumnRetriever
    return ContextBuilder(
        budget,
        kind(top_k=budget.max_columns, concept_top_k=concepts, seed=seed),
        retrieval_threshold=size + 1 if full else 1,
        use_memory=strategy == "semantic_memory",
    )


def main(argv=None):
    """Measure synthetic wide-context behavior, optionally with real LLM evaluation.

    Args:
        argv (sequence[str] or None): Command-line arguments; None reads sys.argv[1:].

    Returns:
        None: Writes strategy configurations, selection logs, a manifest and result table.

    Raises:
        SystemExit: CLI arguments are invalid or help was requested.

    Notes:
        Default mode is offline retrieval-only smoke; actual token/gain metrics remain
        unknown. --llm explicitly enables API spending and inner-CV feature search.
        Held-out rows are evaluated only after all search decisions for a run finish.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[22, 33, 44])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--output", type=Path, default=Path("runs/wide-context"))
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Call the configured LLM and evaluate proposals; default is offline retrieval-only smoke",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-llm-tokens", type=int, default=60_000)
    args = parser.parse_args(argv)
    if min(args.columns) < 2 or args.trials < 1 or args.max_llm_tokens < 1:
        parser.error("Use at least two columns and one trial")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **vars(args),
        "output": str(args.output),
        "evidence_level": "smoke",
        "mode": "llm" if args.llm else "retrieval_only",
        "note": "UTF-8 byte reserves are not billed tokens; retrieval recall is a synthetic oracle proxy. Offline runs provide no proposal/gain evidence.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for size in args.columns:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            X = pd.DataFrame(rng.normal(size=(160, size)), columns=[f"x{i}" for i in range(size)])
            y = X.x0 * X.x1 + rng.normal(scale=0.1, size=len(X))
            schema = DatasetSchema(
                fields=[
                    FieldSchema(
                        name=name,
                        description="loan exposure"
                        if i == 0
                        else "income repayment capacity"
                        if i == 1
                        else f"device measurement {i}",
                        semantic_tags=["credit" if i < 2 else f"device_{i % 20}"],
                    )
                    for i, name in enumerate(X)
                ],
                objective="Predict repayment burden from the interaction of loan exposure and income capacity",
            )
            train, test = train_test_split(np.arange(len(X)), random_state=seed, test_size=0.25)
            for strategy in args.strategies:
                builder = builder_for(strategy, size, seed)
                context = SearchContext(schema, 1, [], [], "rmse", "minimize")
                built = builder.build(
                    context, PROPOSAL_INSTRUCTION, 500_000 if strategy == "full" else 24000, 1
                )
                row = {
                    "columns": size,
                    "seed": seed,
                    "strategy": strategy,
                    "context_columns": len(built.columns),
                    "estimated_prompt_tokens": built.metadata["estimated_prompt_tokens"],
                    "retrieval_latency": built.metadata["retrieval_latency"],
                    "retrieval_recall_proxy": len({"x0", "x1"} & set(built.columns)) / 2,
                    "valid_proposal_rate": None,
                    "prompt_tokens": None,
                    "test_rmse": None,
                    "inner_gain_per_1k_prompt_tokens": None,
                    "inner_gain_per_retrieved_column": None,
                }
                run = args.output / f"{size}-{seed}-{strategy}"
                run.mkdir(exist_ok=True)
                (run / "context.json").write_text(json.dumps(built.metadata, indent=2))
                (run / "configuration.json").write_text(json.dumps(builder.configuration(), indent=2))
                if args.llm:
                    sampler = LLMSampler(
                        context_builder=builder,
                        max_context_chars=500_000 if strategy == "full" else 24000,
                        features_per_trial=1,
                        temperature=0.0,
                    )
                    study = create_study(
                        metric="rmse",
                        direction="minimize",
                        sampler=sampler,
                        storage=run,
                        budget=SearchBudget(max_llm_tokens=args.max_llm_tokens),
                    )
                    study.optimize(
                        X.iloc[train],
                        y.iloc[train],
                        schema,
                        evaluator=CVEvaluator(
                            Ridge(), metric="rmse", cv=3, random_state=seed, importance_repeats=0
                        ),
                        n_trials=args.trials,
                    )
                    # The held-out rows are first accessed after all search decisions finish.
                    score = root_mean_squared_error(
                        y.iloc[test], study.export_pipeline().predict(X.iloc[test])
                    )
                    prompt_tokens = sum(t.tokens.get("input_tokens", 0) for t in study.trials)
                    completion_tokens = sum(t.tokens.get("output_tokens", 0) for t in study.trials)
                    selected_columns = sum(t.context_metadata.get("context_columns", 0) for t in study.trials)
                    gain = study.baseline_trial.value - study.best_value
                    row.update(
                        test_rmse=score,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        valid_proposal_rate=study.search_summary()["valid_proposal_rate"],
                        model_fits=study.consumption["model_fits"],
                        selected_features=len(study.best_features),
                        unknown_requests=study.consumption.get("unknown_requests", 0),
                        stop_reason=study.stop_reason,
                        inner_gain_per_1k_prompt_tokens=gain * 1000 / prompt_tokens
                        if prompt_tokens
                        else None,
                        inner_gain_per_retrieved_column=gain / selected_columns if selected_columns else None,
                    )
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.output / "results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
