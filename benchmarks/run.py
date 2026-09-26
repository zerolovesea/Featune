# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Two-layer, outer-holdout public-data benchmark.

Runs outer-holdout comparisons and preserves failed attempts/resource accounting.
Fresh output directories prevent accidental replacement of prior evidence.

Created:
    2026-09-21
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import CLASSIFICATION, SOURCES, load_dataset
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, get_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import featune
from featune.compiler import build_pipeline
from featune.evaluation import METRICS


def openfe_run(train_x, train_y, test_x, estimator, task, seed, output, candidate_limit):
    """Select row-local OpenFE candidates on outer training rows and fit the downstream model.

    Args:
        train_x (pandas.DataFrame): Outer-training feature rows; all feature selection stays within
            these rows.
        train_y (pandas.Series): Targets aligned with train_x.
        test_x (pandas.DataFrame): Held-out feature rows transformed without fitting statistics on
            them.
        estimator (sklearn estimator): Cloneable prediction model; set its random seed for
            reproducibility.
        task (str): classification or regression, as expected by OpenFE.
        seed (int): Seed for deterministic random choices.
        output (str or Path): Output directory for generated experiment artifacts.
        candidate_limit (int): Maximum OpenFE row-local candidates; zero keeps the entire candidate
            pool.

    Returns:
        tuple: Fitted pipeline, transformed held-out frame, selected-feature count and candidate
        count.

    Notes:
        Statistical OpenFE operators are excluded because its transform implementation
        can pool train/test rows. Writes logs/staging files in output and temporarily
        changes the process working directory; do not run concurrently in one process.
        OpenFE uses its own internal selection budget, not Featune paired-CV accounting.
    """
    from openfe import OpenFE, get_candidate_features, transform, tree_to_formula

    # Official transform concatenates train/test for statistical operators. Restrict to
    # row-local operators so the held-out feature distribution cannot enter training.
    stateless = {
        "abs",
        "log",
        "sqrt",
        "square",
        "sigmoid",
        "round",
        "residual",
        "min",
        "max",
        "+",
        "-",
        "*",
        "/",
    }
    candidates = [
        node
        for node in get_candidate_features(numerical_features=list(train_x.select_dtypes(include="number")))
        if node.name in stateless
    ]
    if candidate_limit:
        indexes = np.random.default_rng(seed).permutation(len(candidates))[:candidate_limit]
        candidates = [candidates[index] for index in indexes]
    train_x, train_y, test_x = (
        train_x.reset_index(drop=True),
        train_y.reset_index(drop=True),
        test_x.reset_index(drop=True),
    )
    # Feather cannot encode mixed string/integer category values such as stock codes.
    for column in train_x.select_dtypes(exclude="number"):
        train_x[column] = train_x[column].astype("string").astype("category")
        test_x[column] = test_x[column].astype("string").astype("category")
    with (output / "openfe.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        selected = OpenFE().fit(
            data=train_x,
            label=train_y.to_frame(),
            task=task,
            candidate_features_list=candidates,
            n_jobs=1,
            seed=seed,
            n_data_blocks=1,
            min_candidate_features=min(20, len(candidates)),
            tmp_save_path=str(output / "openfe.feather"),
            verbose=False,
            stage2_params={"n_estimators": 100, "verbosity": -1, "n_jobs": 1},
        )[:10]
        # OpenFE writes a hard-coded relative Feather file. Keep it inside this run.
        previous_directory = Path.cwd()
        try:
            os.chdir(output.resolve())
            generated_train, generated_test = transform(train_x, test_x, selected, n_jobs=1)
        finally:
            os.chdir(previous_directory)
    (output / "features.json").write_text(json.dumps([tree_to_formula(node) for node in selected], indent=2))
    generated_train.columns = generated_train.columns.astype(str)
    generated_test.columns = generated_test.columns.astype(str)
    schema = featune.DatasetSchema(
        fields=[
            featune.FieldSchema(
                name=n,
                description="OpenFE row-local feature",
                dtype="numeric" if pd.api.types.is_numeric_dtype(generated_train[n]) else "categorical",
            )
            for n in generated_train
        ]
    )
    pipeline = build_pipeline(schema, [], clone(estimator))
    pipeline.fit(generated_train, train_y)
    return pipeline, generated_test, len(selected), len(candidates)


def explain_selected(pipeline, test_x, test_y, names, metric, seed, output):
    """Measure post-selection reliance on derived columns without feeding it back to search.

    Args:
        pipeline: Frozen fitted pipeline selected using training rows only.
        test_x: Outer held-out feature frame.
        test_y: Aligned held-out labels.
        names: Derived columns to permute independently after compilation.
        metric: Featune metric key.
        seed: Reproducible permutation seed.
        output: Directory for the descriptive evidence JSON.

    Returns:
        dict: Explanation runtime, feature count and positive reliance count.

    Notes:
        Three permutations measure predictive reliance, not causality or an independent
        significance test. Descendants are not recomputed; no feature is reselected.
    """
    started = time.monotonic()
    compiled = pipeline["features"].transform(test_x)
    remainder = pipeline[1:]
    scorer = get_scorer(METRICS[metric][0])
    baseline = scorer(remainder, compiled, test_y) if names else None
    rng = np.random.default_rng(seed)
    evidence = {}
    for name in names:
        drops = []
        for _ in range(3):
            permuted = compiled.copy()
            permuted[name] = rng.permutation(compiled[name].to_numpy())
            drops.append(float(baseline - scorer(remainder, permuted, test_y)))
        evidence[name] = {
            "score_drop_mean": float(np.mean(drops)),
            "score_drop_std": float(np.std(drops)),
            "drops": drops,
        }
    (output / "explanations.json").write_text(json.dumps(evidence, indent=2))
    return {
        "posthoc_explanation_seconds": time.monotonic() - started,
        "explained_features": len(evidence),
        "positive_reliance_features": sum(e["score_drop_mean"] > 0 for e in evidence.values()),
    }


def main(argv=None):
    """Run configured public-data holdout comparisons and persist measured evidence.

    Args:
        argv (sequence[str] or None): Command-line arguments; None reads sys.argv[1:].

    Returns:
        None: Writes manifests, per-run artifacts, results.csv and summary.csv.

    Raises:
        SystemExit: CLI arguments are invalid or help was requested.

    Notes:
        Loads/caches public data, may invoke paid APIs for explicitly selected LLM
        methods, and overwrites summaries in the output directory. Selection uses only
        outer-training rows; final test scores are computed after feature selection.
        Use a fresh output directory to preserve old evidence.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=["adult", "california"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "baseline",
            "random",
            "evolution",
            "llm",
            "llm_anonymous",
            "llm_shuffled",
            "hybrid",
            "autonomous",
            "openfe",
        ],
        default=["baseline", "random", "evolution"],
    )
    parser.add_argument("--models", nargs="+", choices=["linear", "tree", "tabpfn"], default=["tabpfn"])
    parser.add_argument(
        "--sample-size", type=int, default=1200, help="Fixed pre-split row cap; 0 keeps full data"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("runs/datasets"))
    parser.add_argument("--importance-repeats", type=int, default=0)
    parser.add_argument(
        "--explain", action="store_true", help="Post-hoc permutation of selected derived columns"
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--max-model-fits", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--openfe-candidates", type=int, default=256, help="0 uses the complete row-local candidate pool"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--report", action="store_true", help="Write per-run HTML reports")
    parser.add_argument("--target-gain", type=float, default=0.005)
    parser.add_argument("--input-price", type=float, default=None, help="Currency per million input tokens")
    parser.add_argument("--output-price", type=float, default=None, help="Currency per million output tokens")
    parser.add_argument("--currency", default="unspecified")
    parser.add_argument("--evidence-level", choices=["smoke", "intermediate"], default="smoke")
    args = parser.parse_args(argv)
    if (
        args.trials < 1
        or args.cv < 2
        or args.openfe_candidates < 0
        or args.sample_size < 0
        or args.importance_repeats < 0
        or (args.max_model_fits is not None and args.max_model_fits < args.cv + 1)
        or not 1 <= args.max_features <= 64
    ):
        parser.error("Invalid budget")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    manifest = {
        **vars(args),
        "output": str(args.output),
        "data_dir": str(args.data_dir),
        "tabpfn_preset": featune.TabPFNClassifier().provenance(),
        "timing": "Sequential runs per process, CPU, imports/checkpoints warmed once before timing; post-hoc explanations timed separately",
        "sampling": "Seeded pre-split subset; paired across models and methods; not a full-data benchmark",
        "sources": SOURCES,
        "evidence_level": args.evidence_level,
        "joint_optimization": False,
        "target_policy": "original targets; no data-dependent clipping",
        "llm": {
            "model": os.getenv("FEATUNE_MODEL", "claude-opus-4-8"),
            "provider": "anthropic",
            "features_per_trial": 1,
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["featune", "numpy", "pandas", "scikit-learn", "tabpfn", "torch", "openfe"]
        },
        "protocol": "25% untouched outer holdout, inner CV search, fixed downstream estimator; OpenFE row-local operators only",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if "tabpfn" in args.models:
        # Exclude one-time checkpoint/import warmup from steady-state method timing.
        warm_x = pd.DataFrame({"x": np.arange(20, dtype=float)})
        for cls, target in [
            (featune.TabPFNClassifier, np.arange(20) % 2),
            (featune.TabPFNRegressor, np.arange(20)),
        ]:
            warm_model = cls(device="cpu", random_state=42).fit(warm_x, target)
            warm_model.predict(warm_x.iloc[:2])
    rows = []
    for dataset in args.datasets:
        base_x, base_y, base_schema = load_dataset(dataset, data_dir=args.data_dir)
        for seed in args.seeds:
            for model in args.models:
                for method in args.methods:
                    classification = dataset in CLASSIFICATION
                    sample_index = np.arange(len(base_x))
                    if args.sample_size and len(sample_index) > args.sample_size:
                        sample_index, _ = train_test_split(
                            sample_index,
                            train_size=args.sample_size,
                            random_state=seed,
                            stratify=base_y if classification else None,
                        )
                        sample_index.sort()
                    X, y = base_x.iloc[sample_index].copy(), base_y.iloc[sample_index].copy()
                    schema = base_schema.model_copy(deep=True)
                    if method == "llm_anonymous":
                        names = {field.name: f"x{i}" for i, field in enumerate(schema.fields)}
                        X = X.rename(columns=names)
                        schema = featune.DatasetSchema(
                            fields=[
                                field.model_copy(
                                    update={
                                        "name": names[field.name],
                                        "description": "Measurement; semantics withheld",
                                    }
                                )
                                for field in schema.fields
                            ],
                            objective="Improve held-out predictive accuracy",
                        )
                    elif method == "llm_shuffled":
                        order = np.random.default_rng(seed).permutation(len(schema.fields))
                        descriptions = {
                            int(a): schema.fields[int(b)].description
                            for a, b in zip(order, np.roll(order, 1))
                        }
                        schema.fields = [
                            field.model_copy(update={"description": descriptions[i]})
                            for i, field in enumerate(schema.fields)
                        ]
                    train_index, test_index = train_test_split(
                        np.arange(len(X)),
                        test_size=0.25,
                        random_state=seed,
                        stratify=y if classification else None,
                    )
                    train_x, train_y = X.iloc[train_index], y.iloc[train_index]
                    test_x, test_y = X.iloc[test_index], y.iloc[test_index]
                    multiclass = classification and y.nunique() > 2
                    metric, direction = (
                        (("auc_ovr" if multiclass else "auc"), "maximize")
                        if classification
                        else ("rmse", "minimize")
                    )
                    estimator = (
                        (LogisticRegression(max_iter=2000, random_state=seed) if classification else Ridge())
                        if model == "linear"
                        else (
                            HistGradientBoostingClassifier(random_state=seed)
                            if classification
                            else HistGradientBoostingRegressor(random_state=seed)
                        )
                    )
                    if model == "tabpfn":
                        cls = featune.TabPFNClassifier if classification else featune.TabPFNRegressor
                        estimator = cls(model_version="v2", device="cpu", n_estimators=4, random_state=seed)
                    run_name = f"{dataset}-{model}-{method}-{seed}"
                    output = args.output / run_name
                    output.mkdir(parents=True, exist_ok=True)
                    started = time.monotonic()
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "split_hash": hashlib.sha256(
                            sample_index[train_index].tobytes() + sample_index[test_index].tobytes()
                        ).hexdigest(),
                        "n_source": len(base_x),
                        "status": "complete",
                        "n_train": len(train_x),
                        "n_test": len(test_x),
                    }
                    study = None
                    try:
                        if method == "openfe":
                            pipeline, test_x, count, candidates = openfe_run(
                                train_x,
                                train_y,
                                test_x,
                                estimator,
                                "classification" if classification else "regression",
                                seed,
                                output,
                                args.openfe_candidates,
                            )
                            row.update(
                                features=count,
                                candidate_count=candidates,
                                tokens=0,
                                trials=None,
                                inner_best=None,
                                candidate_evaluations=None,
                                model_fits=None,
                                time_to_best=None,
                                api_cost=0.0,
                                accounting_note="OpenFE internal screening fits and target-attainment trajectory are not exposed",
                            )
                        else:
                            sampler = featune.RandomSampler(seed=seed)
                            if method == "evolution":
                                sampler = featune.EvolutionSampler(seed=seed)
                            elif method == "autonomous":
                                sampler = featune.AutonomousSampler(
                                    numeric_sampler=featune.RandomSampler(seed=seed)
                                )
                            elif method in {"llm", "llm_anonymous", "llm_shuffled", "hybrid"}:
                                llm = featune.LLMSampler(
                                    provider="anthropic",
                                    max_tokens=2048,
                                    token_budget=40_000,
                                    temperature=0.0,
                                    features_per_trial=1,
                                    input_cost_per_million=args.input_price,
                                    output_cost_per_million=args.output_price,
                                )
                                sampler = (
                                    featune.HybridSampler(llm_sampler=llm, random_sampler=sampler)
                                    if method == "hybrid"
                                    else llm
                                )
                            study = featune.create_study(
                                metric=metric,
                                direction=direction,
                                sampler=sampler,
                                search_strategy="autonomous" if method == "autonomous" else "greedy",
                                max_features=args.max_features,
                                budget=featune.SearchBudget(max_model_fits=args.max_model_fits),
                            )
                            trial_limit = (
                                min(args.trials, (args.max_model_fits - args.cv - 1) // args.cv)
                                if args.max_model_fits is not None
                                else args.trials
                            )
                            study.optimize(
                                train_x,
                                train_y,
                                schema,
                                evaluator=featune.CVEvaluator(
                                    estimator,
                                    metric=metric,
                                    cv=args.cv,
                                    random_state=seed,
                                    importance_repeats=args.importance_repeats,
                                ),
                                n_trials=0 if method == "baseline" else trial_limit,
                            )
                            pipeline = study.export_pipeline()
                            if args.report:
                                study.report(output / "report.html")
                            study.trials_dataframe().to_json(
                                output / "trials.json", orient="records", indent=2
                            )
                            study.export_features(output / "features.json")
                            efficiency = study.search_summary(args.target_gain)
                            (output / "efficiency.json").write_text(
                                json.dumps(efficiency, indent=2), encoding="utf-8"
                            )
                            (output / "fingerprint.json").write_text(
                                study.fingerprint.model_dump_json(indent=2), encoding="utf-8"
                            )
                            study.lineage_dataframe().to_json(
                                output / "lineage.json", orient="records", indent=2
                            )
                            study.pareto_frontier().to_csv(output / "pareto.csv", index=False)
                            row.update(
                                {
                                    key: value
                                    for key, value in efficiency.items()
                                    if not isinstance(value, dict)
                                }
                            )
                            for target_name in ("fixed_gain", "95pct_final_gain"):
                                row.update(
                                    {
                                        target_name + "_" + key: value
                                        for key, value in efficiency[target_name].items()
                                    }
                                )
                            row.update(
                                features=len(study.best_features),
                                inner_best=study.best_value,
                                inner_baseline=study.baseline_trial.value,
                                trials=len(study.trials) - 1,
                                completed=sum(t.state == "COMPLETE" for t in study.trials[1:]),
                                failed=sum(t.state == "FAIL" for t in study.trials),
                                tokens=sum(
                                    t.tokens.get("input_tokens", 0) + t.tokens.get("output_tokens", 0)
                                    for t in study.trials
                                ),
                                unknown_requests=sum(
                                    t.tokens.get("unknown_requests", 0) for t in study.trials
                                ),
                                best_trial=study.best_trial.number,
                            )
                        search_seconds = time.monotonic() - started
                        scorer, sign = METRICS[metric]
                        row.update(
                            test_score=float(get_scorer(scorer)(pipeline, test_x, test_y) * sign),
                            seconds=time.monotonic() - started,
                            search_seconds=search_seconds,
                        )
                        row["holdout_score_seconds"] = row["seconds"] - search_seconds
                        secondary_started = time.monotonic()
                        predicted = pipeline.predict(test_x)
                        if classification:
                            row["test_accuracy"] = float(accuracy_score(test_y, predicted))
                        else:
                            row["test_mae"] = float(mean_absolute_error(test_y, predicted))
                            row["test_r2"] = float(r2_score(test_y, predicted))
                        row["secondary_metric_seconds"] = time.monotonic() - secondary_started
                        if args.explain:
                            names = (
                                list(test_x.columns[-count:])
                                if method == "openfe" and count
                                else (
                                    [feature.name for feature in study.best_features]
                                    if method != "openfe"
                                    else []
                                )
                            )
                            row.update(
                                explain_selected(pipeline, test_x, test_y, names, metric, seed, output)
                            )
                        row["total_seconds_with_explanations"] = time.monotonic() - started
                    except Exception as exc:
                        logging.exception("Benchmark run failed: %s", run_name)
                        row.update(
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                            test_score=None,
                            seconds=time.monotonic() - started,
                            tokens=row.get("tokens"),
                        )
                        if study is not None:
                            study.trials_dataframe().to_json(
                                output / "trials.json", orient="records", indent=2
                            )
                            row["tokens"] = sum(
                                t.tokens.get("input_tokens", 0) + t.tokens.get("output_tokens", 0)
                                for t in study.trials
                            )
                            row["unknown_requests"] = sum(
                                t.tokens.get("unknown_requests", 0) for t in study.trials
                            )
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(args.output / "results.csv", index=False)
                    print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    summary = (
        table[table.status == "complete"]
        .groupby(["dataset", "model", "method"])
        .agg(
            test_mean=("test_score", "mean"),
            test_std=("test_score", "std"),
            runs=("seed", "count"),
            seconds_mean=("seconds", "mean"),
            tokens_total=("tokens", "sum"),
        )
    )
    summary.to_csv(args.output / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
