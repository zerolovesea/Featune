# Benchmark 方法与结果 / Methods and results

Updated: 2026-09-23

当前主实验使用固定 TabPFN v2。历史传统模型、全量数据和离线上下文实验见[历史汇总](results/HISTORICAL.md)，不与新实验混合排名。设计与待办见[对比方案](COMPARISON_PLAN.md)。

本轮新增的自主搜索、真实 LLM 语义检查和大字段消融见[初步结果](results/autonomous-20260923.md)。Wine/Digits 是新外层划分；合成宽表与三项语义检查规模很小，不能据此宣称自主或语义搜索普遍更优。原始逐次记录保留在本地忽略的 `runs/`。

## 保留范围 / Retention

仓库只保留结论、Markdown 汇总对比表、协议/环境说明和报告引用的 PNG。逐次 JSON/CSV、trial/lineage/fingerprint、调试日志、数据库和模型产物已清理。失败、预算阻止和负向结果的数量及限制仍保留在结论中；没有为了改善平均分删去不利运行。

汇总表无法直接重算所有逐次统计。需要复核原始记录时，在被忽略的 `runs/` 下完整重跑，保留本地原始文件；不要用清理后的文档目录作为 summarizer 输入。测试源码、示例配置和示例数据继续维护。

Git contains publication summaries, not raw audit archives. Re-run locally to regenerate raw evidence. The cleanup changes retention, not experimental scores or inclusion criteria.

## 固定 TabPFN v2 协议 / Current protocol

[预声明协议](results/tabpfn-v2/PROTOCOL.md) · [完整报告](results/tabpfn-v2/report.md) · [对比数据](results/tabpfn-v2/comparisons.md) · [环境](results/tabpfn-v2/ENVIRONMENT.md)

- cancer、diabetes、adult、california、covtype、online_retail；每任务最多 1,200 行。
- seeds 11/22/33/44/55，同组共享外层 25% holdout，训练部分 3-fold CV；原始目标不做全量截尾。
- 固定 TabPFN v2、4 个 ensemble、CPU；不搜索模型超参数。
- baseline/random/evolution/语义 LLM/匿名 LLM；Featune 最多四次提议、每次一列，重复、失败或预算终止不追加有利重试。
- OpenFE 0.0.12 限 32 个逐行候选、最多 10 个输出；内部 LightGBM 筛选再用相同 TabPFN，不是完整 OpenFE 或等拟合预算对照。
- 额外使用相同数据划分的原始特征 Linear / HistGradientBoosting，分离模型替换与框架搜索收益。
- 搜索中关闭置换解释；冻结模型后每个入选派生列做三次置换，单独计时，不回流选择。
- 分类 AUC（多分类 weighted OVR）、回归 RMSE，次指标 Accuracy/MAE/R²；五 seed 描述性配对 bootstrap，未校正多重比较。

This is intermediate small-sample evidence. Online Retail uses IID transactions, not temporal/customer-disjoint generalization. Public-data pretraining overlap cannot be excluded. Anonymous prompts hide field metadata, not all domain clues in category values.

## 复现 / Reproduce

使用独立 Python 3.11 环境。先在环境中配置 FEATUNE_API_KEY / FEATUNE_BASE_URL / FEATUNE_MODEL；执行 LLM 对照会产生实际 API 用量。脚本不自动读取 `.env`，不保存凭证。

```bash
python3.11 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
pip install -e '.[benchmark]' 'scikit-learn==1.5.2' 'numpy==1.26.4' 'pandas==2.2.3'
export OMP_NUM_THREADS=2
# Bash: keep every arm on the same dataset/split/search protocol.
common=(--datasets cancer diabetes adult california covtype online_retail
        --seeds 11 22 33 44 55 --trials 4 --cv 3 --sample-size 1200
        --openfe-candidates 32 --explain --evidence-level intermediate)
python benchmarks/run.py "${common[@]}" --models tabpfn \
  --methods baseline random evolution openfe --output runs/tabpfn-v2/core
python benchmarks/run.py "${common[@]}" --models tabpfn \
  --methods llm llm_anonymous --output runs/tabpfn-v2/semantic
python benchmarks/run.py "${common[@]}" --models linear tree \
  --methods baseline --output runs/tabpfn-v2/traditional
python benchmarks/summarize_tabpfn.py runs/tabpfn-v2
```

仅在权重与数据已缓存后，才设置 `HF_HUB_OFFLINE=1`。精确执行环境见上述环境记录；LLM 服务响应不保证跨时间相同。每次用新目录，runner 会重新运行并覆盖所选目录的结果，不提供中途续跑。

`run.py` 默认输出 `runs/benchmark`；`wide.py` 默认输出 `runs/wide-context`。`--report` 生成本地交互 HTML，`--input-price/--output-price` 配置每百万 token 单价，未知价格保持 null。`--sample-size 0` 才是全量输入；TabPFN 容量检查仍然适用。`--openfe-candidates 0` 取消逐行候选池上限，并不启用所有 OpenFE 统计算子。

保留新结论时，将汇总表整理成 Markdown 并更新报告内链接；只提交结论、必要对比表、配置说明和引用的图。不要提交生成的原始运行目录，也不要删除重新执行期间定位失败所需的本地记录。

## 大字段实验 / Wide-context experiments

```bash
python benchmarks/wide.py --columns 200 500 1000 --seeds 22 33 44 --output runs/wide-context-smoke
# Opt-in: real API calls and inner-CV / outer-holdout evaluation.
python benchmarks/wide.py --llm --trials 3 --output runs/wide-llm
```

已完成 45 组离线 smoke，结果保留在[历史汇总](results/HISTORICAL.md)。full/random/semantic/semantic_grouping/semantic_memory 测试的是上下文构建，默认检索为本地字符 TF-IDF；未调用 API 的 token、预测增益和有效提议率记为未测量。真实 LLM 已完成 200/500/1000 字段、五策略、单 seed 初步消融，记录实际 token 与外层分数；多 seed 消融仍待执行，详见[初步结果](results/autonomous-20260923.md)。

## 本轮结论 / Findings


[完整报告](results/tabpfn-v2/report.md) · [汇总对比数据](results/tabpfn-v2/comparisons.md) · [环境记录](results/tabpfn-v2/ENVIRONMENT.md)

2026-09-23 完成 240/240 次运行，无运行失败。这里的成功表示流程完成，不代表搜索改善分数；8 次 LLM 提议受到 token 预算保护而未执行，43 次 Evolution 重复提议复用已有结果。LLM 主实验共记录 1,542,599 token、0 次未知计费请求，另有独立接口探测 73 token；未配置单价，不换算货币成本。

- **准确率收益较小，不能宣传普遍领先。** 语义 LLM 在 Adult 上增加 0.229 个 AUC 百分点，California 配对 RMSE 相对下降 0.927%，Diabetes 下降 0.254%；六个任务的语义 LLM 改善区间均包含零。Online Retail 五次语义搜索为两次退化、三次持平。
- **简单搜索仍有价值。** Adult 的 Random 在五个 seed 全部改善，平均 +0.197 个 AUC 百分点，描述性 95% 区间为 [+0.088, +0.324]。Cancer、Diabetes 上也有小幅正向结果，但多重比较未校正，不能将此表述为普遍显著优势。
- **LLM 语义优势尚未证实。** 六个任务的语义/匿名配对区间均包含零；匿名组在 Diabetes、California、CoverType 的主指标均值更好。仅因有自然语言领域描述，就宣称更准确，不受本次结果支持。
- **效率证据：特征编译开销低，端到端搜索更慢。** 成功 TabPFN CV 的特征编译合计 18.42 秒，占记录的编译/拟合/预测/解释时间 0.201%。语义 LLM 的搜索及主评分耗时约为原始 v2 基线的 3.44–17.70 倍，后验解释另计。OpenFE 搜索本身平均较快，但解释更多列成本较高；两者预算和列数上限不同，不能声称等资源胜出。
- **可解释性的实际能力是可追溯和可核查。** 语义 LLM 选中 39 列，其中 35 列在三次置换的均值上呈正依赖，但这既不是因果证据，也不是解释准确率。California 的某条理由错误地诉诸梯度提升树的机制，土壤编码相邻即有共同地质含义的推测也未经验证；报告保留了这些反例。
- **模型选择仍应由任务决定。** 原始 TabPFN v2 在六个任务中的四个任务取得这组三种原始特征模型的最佳均值；Cancer 是线性模型更好，Online Retail 是 HistGradientBoosting 更好。这是模型层面的比较，不是框架搜索收益。受限 OpenFE 在 California 的平均配对 RMSE 改善为 4.291%，而 CoverType 平均下降 3.308 个 AUC 百分点，不能只引用其中一侧。

本轮支持将 Featune 定位为“可审计、可约束的特征搜索框架”，不支持“LLM 自动带来稳定准确率或速度优势”。后续优化应优先让 LLM 上下文明确下游模型及字段单位/编码含义，核查提议理由，再以独立协议比较相同拟合预算和特征数量下的收益。不要依据本轮 holdout 挑选更好的提示词后继续复用该 holdout 宣称泛化提升。

All 240 runs completed. Semantic LLM improvements are small and their descriptive intervals include zero on every task; semantic-versus-anonymous intervals also include zero. Adult Random improved all five seeds. Feature compilation accounted for 0.201% of recorded successful CV time, but semantic LLM search plus primary scoring cost 3.44–17.70 times the raw v2 baseline. Eight proposals were budget-blocked, and 43 evolutionary duplicates reused results. The measured value is traceable, constrained search, not universal accuracy, speed, or explanation correctness. The raw v2 model led four of six task means among the three model controls; linear won Cancer and HistGradientBoosting won Online Retail. Model gains must not be attributed to feature search.
