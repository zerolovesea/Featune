# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Self-contained, escaped interactive experiment reports.

Produces self-contained Plotly HTML with escaped user-facing text. Importing
Plotly is included by default and imported only when a report is requested.

Created:
    2026-09-21
"""

from html import escape
from pathlib import Path


def render_report(study, path=None):
    """Render escaped trial evidence and interactive charts as standalone HTML.

    Args:
        study (FeatureStudy): Study with a completed baseline and at least one completed trial.
        path (str, Path, or None): Optional destination; missing parent directories are created.

    Returns:
        str: Complete HTML document with bundled Plotly JavaScript.

    Raises:
        ValueError: The study has no completed trial.
        ImportError: The default Plotly dependency is missing from a broken installation.

    Notes:
        Writes the destination when requested and otherwise supports notebook HTML
        display. User-controlled labels/tables are escaped. Metric trajectories are
        inner-CV search evidence, not independent test or causal claims.
    """
    import plotly.graph_objects as go

    completed = [trial for trial in study.trials if trial.state == "COMPLETE"]
    best = study.best_trial
    titles = ["Validation trajectory", "LLM token usage", "Permutation importance", "Trial duration"]
    figures = [go.Figure() for _ in titles]
    figures[0].add_trace(
        go.Scatter(
            x=[t.number for t in completed],
            y=[t.value for t in completed],
            mode="lines+markers",
            name=study.metric,
        )
    )
    for token_type in ("input_tokens", "output_tokens"):
        figures[1].add_trace(
            go.Bar(
                x=[t.number for t in study.trials],
                y=[t.tokens.get(token_type, 0) for t in study.trials],
                name=token_type,
            )
        )
    importance = sorted(best.importance.items(), key=lambda item: item[1])[-20:]
    figures[2].add_trace(
        go.Bar(
            x=[value for _, value in importance],
            y=[escape(name) for name, _ in importance],
            orientation="h",
            name="Held-out permutation",
        )
    )
    figures[3].add_trace(
        go.Scatter(
            x=[t.number for t in study.trials],
            y=[t.duration for t in study.trials],
            mode="lines+markers",
            name="Seconds",
        )
    )
    summary = study.search_summary()
    curve = study.pareto_frontier()
    titles.append("Pareto frontier · candidate evaluations")
    figures.append(
        go.Figure(go.Scatter(x=curve.get("evaluations", []), y=curve.get("value", []), mode="lines+markers"))
    )
    titles.append("FeatureSet evolution")
    figures.append(
        go.Figure(
            go.Scatter(
                x=[t.number for t in completed], y=[t.feature_count for t in completed], mode="lines+markers"
            )
        )
    )
    charts = []
    for index, (figure, title) in enumerate(zip(figures, titles)):
        figure.update_layout(
            template="plotly_white",
            height=400,
            autosize=True,
            title={"text": title, "font": {"size": 18}},
            showlegend=index == 1,
            legend={"orientation": "h", "y": -0.2},
            font={"family": "Arial, sans-serif"},
            colorway=["#26756d", "#d28a47", "#435c86"],
            margin={"t": 60, "l": 60, "r": 24, "b": 60},
            paper_bgcolor="#faf9f6",
            plot_bgcolor="#faf9f6",
            barmode="stack",
        )
        figure.update_yaxes(automargin=True)
        charts.append(
            '<div class="chart" role="img" aria-label="'
            + title
            + '">'
            + figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"responsive": True, "displaylogo": False},
            )
            + "</div>"
        )
    history = study.feature_history()
    proposals = study.proposal_history()
    trials = study.trials_dataframe()
    columns = [
        name
        for name in [
            "number",
            "state",
            "value",
            "delta",
            "parent",
            "parent_delta",
            "feature_count",
            "model_fits",
            "transform_time",
            "training_time",
            "prediction_time",
            "explanation_time",
            "cache_hit",
            "duration",
            "error",
        ]
        if name in trials
    ]
    total_tokens = sum(
        t.tokens.get("input_tokens", 0) + t.tokens.get("output_tokens", 0) for t in study.trials
    )
    import pandas as pd

    resources = pd.DataFrame(
        [
            {"Metric": key, "Value": "Unknown" if value is None else value}
            for key, value in summary.items()
            if not isinstance(value, dict)
        ]
        + [{"Metric": "stop_reason", "Value": study.stop_reason or "Unknown"}]
    )
    provenance = getattr(getattr(study, "fingerprint", None), "model_provenance", {})
    attribution = "<p>Built with PriorLabs-TabPFN</p>" if provenance else ""
    if provenance.get("model") == "TabPFN-3.5":
        attribution += (
            "<p>TabPFN-3.5 weights and outputs are subject to the "
            '<a href="https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/'
            '06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/LICENSE">'
            "Prior Labs non-commercial license</a>. Production, commercial and "
            "hosted use (including free services) require separate permission.</p>"
        )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Featune — {escape(study.study_name)}</title>
<style>body{{margin:0;background:#faf9f6;color:#202b29;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:44px 24px}}h1{{font-size:42px;letter-spacing:-1.5px;margin:0}}
small{{letter-spacing:2px;color:#26756d}}.summary{{display:flex;gap:48px;flex-wrap:wrap;padding:25px 0;border-block:1px solid #d8ddd8}}
.summary strong{{display:block;font-size:28px}}section{{overflow:auto;margin:32px 0}}table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:13px}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #d8ddd8;max-width:420px;overflow-wrap:anywhere}}th{{white-space:nowrap}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:24px;margin-top:32px}}.chart{{min-width:0}}th{{background:#eef1ec}}.note{{color:#59645f}}button{{padding:8px 14px;background:#26756d;color:white;border:0;cursor:pointer}}
</style></head><body><main><small>FEATUNE / EXPERIMENT REPORT</small>
{attribution}<h1>{escape(study.study_name)}</h1><p>Semantic feature engineering · {escape(study.metric)} · {escape(study.direction)}</p>
<div class="summary"><div>Best validation score<strong>{best.value:.6f}</strong></div>
<div>Baseline<strong>{study.baseline_trial.value:.6f}</strong></div><div>Selected features<strong>{len(best.features)}</strong></div>
<div>Reported tokens<strong>{total_tokens:,}</strong></div></div>
<div class="charts">{"".join(charts)}</div>
<p class="note">Scores are inner cross-validation estimates used for search, not unbiased test performance.
Directions are Spearman associations, not SHAP. Hypotheses describe joint feature-set changes and are not causal claims.
Token totals exclude requests with unknown billing; inspect trial records for uncertainty.</p>
<section><h2>Search efficiency and resources</h2>{resources.to_html(index=False, escape=True, na_rep="Unknown")}</section>
<section><h2>Feature lineage</h2>{study.lineage_dataframe().to_html(index=False, escape=True)}</section>
<section><h2>Concept evidence</h2>{study.concept_summary().to_html(index=False, escape=True)}</section>
<section><h2>Feature hypotheses</h2>{history.to_html(index=False, escape=True)}</section>
<section><h2>LLM / sampler proposals</h2>{proposals.to_html(index=False, escape=True, na_rep="Unknown")}</section>
<section><h2>All trials</h2>{trials[columns].to_html(index=False, escape=True)}</section>
</main><script>
const charts = document.querySelectorAll('.js-plotly-plot');
const resize = new ResizeObserver(entries => entries.forEach(entry => Plotly.Plots.resize(entry.target)));
charts.forEach(chart => resize.observe(chart));
</script></body></html>"""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html
