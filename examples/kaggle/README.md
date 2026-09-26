# Featune on Kaggle Playground

The notebooks below show a complete, public-data workflow: install Featune, load the official competition CSVs, declare a `DatasetSchema`, compare a baseline with controlled feature proposals using the same folds, inspect trials, check an untouched holdout, refit the selected pipeline on **all labeled rows**, and write a validated `submission.csv`.

| Episode | Task | Metric | Notebook |
|---|---|---|---|
| [S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) | Electric vehicle purchase prediction | ROC AUC | [Source](s6e9/s6e9.ipynb) · [Kaggle](https://www.kaggle.com/code/yaaangzhou/featune-ev-purchase-full-data-training) |
| [S6E8](https://www.kaggle.com/competitions/playground-series-s6e8) | Smartphone addiction prediction | ROC AUC | [Source](s6e8/s6e8.ipynb) · [Kaggle](https://www.kaggle.com/code/yaaangzhou/featune-smartphone-addiction-full-data-training) |

The default TabPFN v2 model has a 10,000-row training limit. These competitions have roughly 669,000 and 691,000 labeled rows, so both notebooks explicitly pass a LightGBM classifier through `CVEvaluator(estimator=...)`. Featune remains responsible for the feature proposal, compilation, paired-fold evaluation, evidence and exported pipeline. A selected feature is retained only if it improves the inner cross-validation score; the baseline may win.

## Run locally

Join each competition and accept its rules on Kaggle first. Then, from the repository root:

```bash
pip install -e . lightgbm nbclient ipykernel
mkdir -p runs/kaggle/s6e9 runs/kaggle/s6e8
kaggle competitions download playground-series-s6e9 -p runs/kaggle/s6e9
kaggle competitions download playground-series-s6e8 -p runs/kaggle/s6e8
unzip -o runs/kaggle/s6e9/playground-series-s6e9.zip -d runs/kaggle/s6e9
unzip -o runs/kaggle/s6e8/playground-series-s6e8.zip -d runs/kaggle/s6e8
jupyter lab examples/kaggle/s6e9/s6e9.ipynb
```

Run S6E8 by opening its notebook similarly. Locally generated submissions stay under ignored `runs/kaggle/`. On Kaggle, the competition source is attached by the notebook metadata and `submission.csv` is written to `/kaggle/working/`. The notebooks install `featune==1.0.0` only when Featune is absent; internet must be enabled for that first install.

S6E9 remains open through 2026-09-30 23:59 UTC. S6E8 closed on 2026-08-31 and accepts **late submissions**, which do not change the original competition standings. Competition data are not committed or redistributed here.

## API map / API 速览

| API | Role / 作用 |
|---|---|
| `FieldSchema` / `DatasetSchema` | Describe allowed columns, types and prediction semantics / 声明可用字段、类型与目标 |
| `RandomSampler` | Propose reproducible controlled expressions / 提出可重复的受控特征 |
| `CVEvaluator` | Supply the metric, model and paired folds / 指定指标、模型与相同的验证折 |
| `create_study` / `optimize` | Search and retain baseline or a better feature set / 搜索并保留 baseline 或更优特征集 |
| `trials_dataframe` / `best_features` | Audit every attempt and inspect the selected set / 检查各次试验及最终特征 |
| `export_pipeline` | Obtain the fitted sklearn pipeline; clone and fit it on all rows for submission / 导出 pipeline，再克隆并用全部训练行拟合 |

The inner CV score and local holdout AUC are different from Kaggle's leaderboard score. The two-trial budget here demonstrates the workflow; it does not establish statistical superiority. No LLM key is required or embedded in a public notebook.

| Episode | Baseline / best inner CV AUC | Untouched holdout AUC | Kaggle score |
|---|---:|---:|---:|
| S6E9 | 0.940597 / 0.940597 | 0.941203 | Public 0.94049 |
| S6E8 (late) | 0.948603 / 0.948603 | 0.947676 | Public 0.94921 · private 0.95027 |

Both searches retained the raw-field baseline; the two controlled proposals in each run did not improve inner CV. S6E8 was submitted after the official close and does not affect the original rankings. These are single-split, single-seed example results, not comparative performance claims.
