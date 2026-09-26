# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Summarize the paired TabPFN v2 comparison without selecting favorable seeds.

Created:
    2026-09-22
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["random", "evolution", "llm", "llm_anonymous", "openfe"]


def interval(values):
    """Return a seeded paired-mean bootstrap interval across independent split seeds.

    Args:
        values (array-like): One directional improvement per paired seed.

    Returns:
        tuple: Mean, 2.5th and 97.5th percentiles; NaN bounds for fewer than two seeds.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(np.mean(values)), float("nan"), float("nan")
    rng = np.random.default_rng(20260922)
    means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def main(argv=None):
    """Write complete-run tables, paired deltas, an explanatory chart and a Chinese report.

    Args:
        argv (sequence[str] or None): CLI options; reads process arguments when absent.

    Returns:
        None: Writes report.md, comparison.png and auditable CSV tables under output.

    Raises:
        ValueError: Duplicate or unmatched split identities would invalidate comparison.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    frames = []
    for part in ["core", "semantic", "traditional"]:
        frame = pd.read_csv(args.output / part / "results.csv")
        manifest = json.loads((args.output / part / "manifest.json").read_text())
        expected = set(
            itertools.product(*(manifest[key] for key in ["datasets", "models", "methods", "seeds"]))
        )
        observed = set(frame[["dataset", "model", "method", "seed"]].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"Incomplete or unexpected run matrix in {part}; finish all planned runs first")
        frame["part"] = part
        frames.append(frame)
    all_runs = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "model", "method", "seed"]
    if all_runs.duplicated(keys).any():
        raise ValueError("Duplicate runs; never choose the most favorable duplicate")
    if all_runs.groupby(["dataset", "seed"]).split_hash.nunique().max() != 1:
        raise ValueError("Methods do not share the same outer split")
    all_runs.to_csv(args.output / "all_runs.csv", index=False)
    complete = all_runs[all_runs.status == "complete"].copy()
    summary = (
        complete.groupby(["dataset", "model", "method"])
        .agg(
            score_mean=("test_score", "mean"),
            score_std=("test_score", "std"),
            runs=("seed", "count"),
            seconds_mean=("seconds", "mean"),
            explanation_seconds_mean=("posthoc_explanation_seconds", "mean"),
            features_mean=("features", "mean"),
            tokens_total=("tokens", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    complete.groupby(["dataset", "model", "method"])[["test_accuracy", "test_mae", "test_r2"]].agg(
        ["mean", "std"]
    ).to_csv(args.output / "secondary_metrics.csv")
    all_runs.groupby("dataset").agg(
        source_rows=("n_source", "first"),
        training_rows=("n_train", "first"),
        test_rows=("n_test", "first"),
        metric=("metric", "first"),
    ).to_csv(args.output / "datasets.csv")
    breakdown = []
    proposal_states = []
    for row in complete[complete.method != "openfe"].itertuples():
        path = args.output / row.part / f"{row.dataset}-{row.model}-{row.method}-{row.seed}" / "trials.json"
        trials = json.loads(path.read_text())
        measured = [trial for trial in trials if trial["state"] == "COMPLETE"]
        record = {"dataset": row.dataset, "model": row.model, "method": row.method, "seed": row.seed}
        proposal_states.extend(
            {**record, "number": trial["number"], "state": trial["state"]}
            for trial in trials
            if trial["number"] != 0
        )
        for key in ["transform_time", "training_time", "prediction_time", "explanation_time"]:
            record[key] = sum(trial.get(key, 0) for trial in measured)
        breakdown.append(record)
    breakdown = pd.DataFrame(breakdown)
    breakdown.to_csv(args.output / "cv_timing_breakdown.csv", index=False)
    proposals = pd.DataFrame(proposal_states)
    proposals.to_csv(args.output / "proposal_states.csv", index=False)
    baseline = complete[(complete.model == "tabpfn") & (complete.method == "baseline")]
    checks = complete[(complete.model == "tabpfn") & (complete.method != "openfe")].merge(
        baseline[["dataset", "seed", "inner_best"]],
        on=["dataset", "seed"],
        suffixes=("", "_reference"),
        validate="many_to_one",
    )
    if not np.allclose(checks.inner_baseline, checks.inner_best_reference, rtol=1e-6, atol=1e-8):
        raise ValueError("Raw-model inner CV differs across methods; audit before interpreting semantics")
    paired = complete.merge(
        baseline[["dataset", "seed", "test_score", "seconds"]],
        on=["dataset", "seed"],
        suffixes=("", "_baseline"),
        validate="many_to_one",
    )
    paired["improvement"] = np.where(paired.metric == "rmse", -1, 1) * (
        paired.test_score - paired.test_score_baseline
    )
    paired["improvement_display"] = (
        np.where(paired.metric == "rmse", paired.improvement / paired.test_score_baseline, paired.improvement)
        * 100
    )
    paired["time_ratio"] = paired.seconds / paired.seconds_baseline
    paired.to_csv(args.output / "paired_runs.csv", index=False)
    deltas = []
    for (dataset, model, method), group in paired.groupby(["dataset", "model", "method"]):
        mean, lo, hi = interval(group.improvement_display)
        deltas.append(
            dict(
                dataset=dataset,
                model=model,
                method=method,
                mean=mean,
                ci_low=lo,
                ci_high=hi,
                pairs=len(group),
                wins=int((group.improvement > 1e-10).sum()),
                ties=int((group.improvement.abs() <= 1e-10).sum()),
                time_ratio_mean=group.time_ratio.mean(),
            )
        )
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(args.output / "paired_summary.csv", index=False)
    semantic = complete[(complete.model == "tabpfn") & (complete.method == "llm")].merge(
        complete[(complete.model == "tabpfn") & (complete.method == "llm_anonymous")],
        on=["dataset", "seed"],
        suffixes=("", "_anonymous"),
        validate="one_to_one",
    )
    semantic["semantic_improvement"] = np.where(semantic.metric == "rmse", -1, 1) * (
        semantic.test_score - semantic.test_score_anonymous
    )
    semantic.to_csv(args.output / "semantic_pairs.csv", index=False)

    datasets = list(dict.fromkeys(all_runs.dataset))
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, dataset in zip(axes.flat, datasets):
        data = (
            deltas[(deltas.dataset == dataset) & (deltas.model == "tabpfn")]
            .set_index("method")
            .reindex(METHODS)
        )
        means = data["mean"].to_numpy()
        ax.barh(METHODS, means, color=["#32786c" if x >= 0 else "#b56859" for x in means])
        ax.errorbar(
            means,
            np.arange(len(METHODS)),
            xerr=[means - data.ci_low, data.ci_high - means],
            fmt="none",
            color="#333333",
            capsize=3,
        )
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(dataset)
        metric = all_runs.loc[all_runs.dataset == dataset, "metric"].iloc[0]
        ax.set_xlabel(
            "Relative RMSE reduction (%)" if metric == "rmse" else "AUC improvement (percentage points)"
        )
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Feature search vs raw TabPFN v2 — paired five-seed results\nPositive favors search; descriptive 95% bootstrap intervals",
        fontsize=13,
    )
    fig.supxlabel(
        "Up to 1,200 rows · 4 search trials · OpenFE: 32 row-local candidates · Built with PriorLabs-TabPFN",
        fontsize=9,
    )
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)

    lines = [
        "# TabPFN v2 多数据集基准结果",
        "",
        "Built with PriorLabs-TabPFN",
        "",
        "实验协议见 [PROTOCOL.md](PROTOCOL.md)，全部运行见 [all_runs.csv](all_runs.csv)，Accuracy/MAE/R² 见 [次指标汇总](secondary_metrics.csv)。",
        "结果解读与工程建议见 [benchmark 文档](../../README.md)。",
        "",
        f"共记录 {len(all_runs)} 次运行；成功 {len(complete)}，失败 {len(all_runs) - len(complete)}。失败不删除，不用有利 seed 补位。",
        "",
        "## 外层测试结果",
        "",
        "AUC 越高越好，RMSE 越低越好；每格为均值 ± 样本标准差。不同数据集数值不能直接横比。",
        "",
        "| 数据集 | 原始 v2 | Random | Evolution | 语义 LLM | 匿名 LLM | OpenFE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        metric_name = complete.loc[complete.dataset == dataset, "metric"].iloc[0]
        cells = [f"{dataset} ({metric_name})"]
        for method in ["baseline", *METHODS]:
            rows = summary[
                (summary.dataset == dataset) & (summary.model == "tabpfn") & (summary.method == method)
            ]
            cells.append(
                "缺失" if rows.empty else f"{rows.iloc[0].score_mean:.6f} ± {rows.iloc[0].score_std:.6f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 模型替换与框架收益分离",
        "",
        "下表全部使用原始特征；此处的收益属于下游模型变化，不属于 Featune 特征搜索。",
        "",
        "| 数据集 | Linear | HistGradientBoosting | TabPFN v2 |",
        "|---|---:|---:|---:|",
    ]
    for dataset in datasets:
        cells = [dataset]
        for model in ["linear", "tree", "tabpfn"]:
            data = summary[
                (summary.dataset == dataset) & (summary.model == model) & (summary.method == "baseline")
            ]
            cells.append(
                "缺失" if data.empty else f"{data.iloc[0].score_mean:.6f} ± {data.iloc[0].score_std:.6f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 相对原始 v2 的配对改善",
        "",
        "分类单位为 AUC 百分点；回归为 RMSE 相对下降百分比。正值表示改善。区间仅基于 5 个随机划分，未作多重比较校正。",
        "",
        "| 数据集 | 方法 | 改善均值 [95% CI] | 胜/平/配对数 | 搜索及评分耗时倍数 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in deltas[(deltas.model == "tabpfn") & (deltas.method != "baseline")].itertuples():
        lines.append(
            f"| {row.dataset} | {row.method} | {row.mean:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {row.wins}/{row.ties}/{row.pairs} | {row.time_ratio_mean:.2f}× |"
        )
    lines += ["", "![配对改善](comparison.png)", "", "## 语义、效率与解释", ""]
    for dataset, group in semantic.groupby("dataset"):
        mean, lo, hi = interval(group.semantic_improvement)
        lines.append(
            f"- {dataset}：正确语义相对匿名语义的方向统一原始指标改善 {mean:+.6f}，区间 [{lo:+.6f}, {hi:+.6f}]；{len(group)} 对。"
        )
    lines += [
        "",
        "| 方法 | 平均搜索及评分秒数 | 平均后验解释秒数 | 平均派生列数 | 正依赖列/解释列 | 已知 token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, group in complete[complete.model == "tabpfn"].groupby("method"):
        positive = int(group.positive_reliance_features.fillna(0).sum())
        explained = int(group.explained_features.fillna(0).sum())
        lines.append(
            f"| {method} | {group.seconds.mean():.2f} | {group.posthoc_explanation_seconds.mean():.2f} | {group.features.mean():.2f} | {positive}/{explained} | {int(group.tokens.fillna(0).sum()):,} |"
        )
    lines += [
        "",
        "上表按完整数据集矩阵平均，需同时查看各数据集明细；正依赖数量受候选列数、相关性和采样误差影响，不是解释质量排名。Featune 最多新增 4 列、OpenFE 最多 10 列，列数上限不同，因此不能仅凭列数较少宣称算法更稀疏。",
    ]
    lines += [
        "",
        "成功运行的提议状态见 [proposal_states.csv](proposal_states.csv)。预算是提议次数，重复提议不重新拟合，因此不同方法的实际拟合数可能不同：",
        "",
        "| 方法 | 提议状态 | 次数 |",
        "|---|---|---:|",
    ]
    for (method, state), count in proposals.groupby(["method", "state"]).size().items():
        lines.append(f"| {method} | {state} | {count} |")
    lines += [
        "",
        f"LLM 已知 token 合计：{int(all_runs.tokens.fillna(0).sum()):,}；未知计费请求：{int(all_runs.get('unknown_requests', pd.Series(dtype=float)).fillna(0).sum())}。未配置单价，不能报告准确货币成本。",
        "",
        "可解释性证据包括 features.json 的表达式/假设、trials.json 的 CV 轨迹、lineage.json 的来源及 explanations.json 的最终派生列置换结果。置换只描述冻结模型对该列的依赖，不是因果证明；搜索中未开启置换，因此不能把 inconclusive 假设改写为已证实。",
    ]
    lines += [
        "",
        "### 可解释性实例",
        "",
        "固定展示各数据集最小 seed 的语义 LLM 运行中，导出顺序的第一条入选表达式，不挑最大收益或最高重要性。完整表达式与理由见每个运行目录。",
        "",
        "| 数据集 / seed | 表达式 | LLM 提出的概念 | 置换分数下降均值 |",
        "|---|---|---|---:|",
    ]
    for dataset in datasets:
        runs = complete[
            (complete.dataset == dataset) & (complete.model == "tabpfn") & (complete.method == "llm")
        ].sort_values("seed")
        if runs.empty:
            continue
        row = runs.iloc[0]
        directory = f"semantic/{dataset}-tabpfn-llm-{row.seed}"
        features = json.loads((args.output / directory / "features.json").read_text())["features"]
        label = f"[{dataset} / {row.seed}]({directory}/features.json)"
        if not features:
            lines.append(f"| {label} | 未新增特征 | 保留原始模型 | — |")
            continue
        feature = features[0]
        expression = f"{feature['op']}({', '.join(feature['inputs'])})"
        concept = feature["hypothesis"]["concept"].replace("|", "\\|").replace("\n", " ")
        evidence = json.loads((args.output / directory / "explanations.json").read_text())
        drop = evidence[feature["name"]]["score_drop_mean"]
        lines.append(f"| {label} | `{expression}` | {concept} | {drop:+.6f} |")
    lines += [
        "",
        "置换下降使用原始评分单位：AUC 下降或 RMSE 上升，不能跨数据集比较；单列依赖也不等于新增该列相对原始模型的净收益。LLM 的领域理由需要人工核查。例如 California/seed11 的理由谈及梯度提升树的轴对齐切分，而实际下游模型是 TabPFN；CoverType/seed11 对相邻土壤编码存在共同地质含义的推测，也没有在本实验中验证。可追溯不代表解释必然正确。",
        "",
        "## 适用范围与限制",
        "",
        "- 使用最多 1,200 行子样本，不能声称在完整 CoverType/Online Retail 上取得同等收益；Online Retail 是 IID 交易划分，不代表时间或客户外推。",
        "- 原始 v2 基线也运行 3-fold CV 和最终 refit，其耗时不是单次裸模型 fit 成本。OpenFE 使用自己的 LightGBM 筛选预算，内部拟合数未知，不能声称严格资源公平。",
        "- 耗时使用 time.monotonic() 差值，包含搜索、refit、输出工件和一次主指标评分；不包含数据下载和预热。次指标和后验解释单独计时，不把整项任务的实际起止历时当成算法耗时。方法顺序固定、预热一次，系统负载仍会带来波动。",
        "- v2 分类权重经过上游微调，公共数据预训练重合风险未排除；这不是无污染的新任务泛化证明。",
        "- 匿名组只隐藏列名、描述和任务目标，保留数值及类别取值；这是字段元数据消融，不保证 LLM 完全无法识别领域或数据集。",
        "- 无选择性删除失败或只展示获胜任务；小预算、五 seed 的结果不能证明普遍领先。",
        "",
        "## English summary",
        "",
        "Paired six-dataset, five-seed small-sample comparison using pinned TabPFN v2. Positive deltas favor feature search. All failures and neutral/negative results are retained. Bootstrap intervals are descriptive; OpenFE budgets differ. Feature lineage and permutation reliance are audit evidence, not causal explanations.",
    ]
    tabpfn_timing = breakdown[breakdown.model == "tabpfn"]
    transform_seconds = tabpfn_timing.transform_time.sum()
    measured_seconds = (
        tabpfn_timing[["transform_time", "training_time", "prediction_time", "explanation_time"]].sum().sum()
    )
    lines += [
        "",
        "## 成功 CV 的耗时拆解",
        "",
        f"TabPFN 对照中，成功 CV 的特征编译合计 {transform_seconds:.2f} 秒，占已记录编译/拟合/预测/解释时间的 {100 * transform_seconds / measured_seconds:.3f}%。详见 [分项计时](cv_timing_breakdown.csv)。此口径不含 LLM、失败尝试或最终 refit，不能替代端到端速度比较。",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "counts.json").write_text(
        json.dumps(
            {"runs": len(all_runs), "complete": len(complete), "failed": len(all_runs) - len(complete)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
