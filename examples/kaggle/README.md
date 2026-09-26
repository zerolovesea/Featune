# Featune on Kaggle Playground

The notebooks below show a complete, public-data workflow: install Featune, load the official competition CSVs, declare a `DatasetSchema`, compare a baseline with controlled feature proposals using the same folds, inspect trials, check an untouched holdout, refit the selected pipeline on a **stratified sample within TabPFN v2's limit**, and write a validated `submission.csv` for every test row.

| Episode | Task | Metric | Notebook |
|---|---|---|---|
| [S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) | Electric vehicle purchase prediction | ROC AUC | [Source](s6e9/s6e9.ipynb) · [Kaggle](https://www.kaggle.com/code/yaaangzhou/featune-ev-purchase-with-tabpfn-v2) |
| [S6E8](https://www.kaggle.com/competitions/playground-series-s6e8) | Smartphone addiction prediction | ROC AUC | [Source](s6e8/s6e8.ipynb) · [Kaggle](https://www.kaggle.com/code/yaaangzhou/featune-smartphone-addiction-tabpfn-v2) |

The default TabPFN v2 model has a 10,000-row single-fit limit with an accelerator and 1,000 rows on CPU. These competitions have roughly 669,000 and 691,000 labeled rows, so both notebooks reserve a 2,000-row holdout, stratify a device-sized training sample, search on a smaller subset, then predict the entire official test set in batches. They use `CVEvaluator` without an estimator override, so the default model is TabPFN v2. A selected feature is retained only if it improves inner cross-validation; the baseline may win. The sampled fit does **not** use every labeled row.

## Run locally

Join each competition and accept its rules on Kaggle first. Then, from the repository root:

```bash
pip install -e . nbclient ipykernel
mkdir -p runs/kaggle/s6e9 runs/kaggle/s6e8
kaggle competitions download playground-series-s6e9 -p runs/kaggle/s6e9
kaggle competitions download playground-series-s6e8 -p runs/kaggle/s6e8
unzip -o runs/kaggle/s6e9/playground-series-s6e9.zip -d runs/kaggle/s6e9
unzip -o runs/kaggle/s6e8/playground-series-s6e8.zip -d runs/kaggle/s6e8
jupyter lab examples/kaggle/s6e9/s6e9.ipynb
```

Run S6E8 by opening its notebook similarly. Locally generated submissions stay under ignored `runs/kaggle/`. On Kaggle, the competition source is attached by the notebook metadata and `submission.csv` is written to `/kaggle/working/`. S6E9 installs `featune==1.0.0` and S6E8 installs `featune==1.1.0` only when Featune is absent; internet must be enabled for that first install.

S6E9 remains open through 2026-09-30 23:59 UTC. S6E8 closed on 2026-08-31 and accepts **late submissions**, which do not change the original competition standings. Competition data are not committed or redistributed here.

## API map / API 速览

| API | Role / 作用 |
|---|---|
| `FieldSchema` / `DatasetSchema` | Describe allowed columns, types and prediction semantics / 声明可用字段、类型与目标 |
| `RandomSampler` | Propose reproducible controlled expressions / 提出可重复的受控特征 |
| `CVEvaluator` | Supply the metric and paired folds; omitted estimator selects TabPFN v2 / 指定指标与验证折，默认选择 TabPFN v2 |
| `create_study` / `optimize` | Search and retain baseline or a better feature set / 搜索并保留 baseline 或更优特征集 |
| `trials_dataframe` / `best_features` | Audit every attempt and inspect the selected set / 检查各次试验及最终特征 |
| `export_pipeline` | Obtain the fitted sklearn pipeline; clone and fit it on the allowed sample / 导出 pipeline，再克隆并在允许的样本上拟合 |

The inner CV score and local holdout AUC are different from Kaggle's leaderboard score. The two-trial budget here demonstrates the workflow; it does not establish statistical superiority. No LLM key is required or embedded in a public notebook.

| Episode | Final fit rows | Baseline / best inner CV AUC | Untouched holdout AUC | Kaggle score |
|---|---:|---:|---:|---:|
| S6E9 | 10,000 | 0.941619 / 0.941619 | 0.942109 | Public 0.93720 |
| S6E8 (late, original) | 10,000 | 0.918793 / 0.922253 | 0.923583 | Public 0.93614 · private 0.93723 |
| S6E8 (late, screen-time reconstruction + missing flags) | 10,000 | 0.921913 / 0.922825 | 0.924915 | Public 0.93743 · private 0.93846 |

These are single-split, single-seed example results. S6E9 retained the raw-field baseline; S6E8 selected one controlled multiplication feature. The S6E8 source now also includes six screen-time ratios; its full Kaggle run is pending, so no leaderboard score is claimed for that revision. Earlier full-data LightGBM scores are excluded because they are not results from these notebooks. S6E8 is closed and accepts late submissions, which do not change the original competition standings.
