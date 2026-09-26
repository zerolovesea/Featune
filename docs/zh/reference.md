# DSL 与 API 参考

[English](../en/reference.md) · [指南](guide.md)

## 受控 DSL

```json
{
  "features": [
    {
      "name": "debt_income_ratio",
      "op": "divide",
      "inputs": ["loan", "income"],
      "params": {},
      "hypothesis": {
        "concept": "repayment_capacity",
        "rationale": "本金相对月可支配收入近似衡量偿债负担",
        "expected_direction": "increasing"
      }
    }
  ],
  "remove": [],
  "model_params": {"C": 1.0}
}
```

`remove` 可列出要删除的父集合派生特征名；其依赖者必须一并删除。`model_params` 必须与 study 的 param_space 中允许的值及类型一致；未启用参数搜索时应为 `{}`。特征名以英文字母开头，只能包含 ASCII 字母、数字、下划线，最多 100 字符。原始字段名可使用中文或空格。每个特征必须有 concept/rationale；方向为 increasing/decreasing/unknown。依赖必须是可用原始字段或列表中更早的特征，不允许覆盖原始字段、重复表达式或前向引用。所有 schema 拒绝多余键。

| op | 输入 | 输出/定义 |
|---|---|---|
| add / subtract / multiply | 两个 numeric | 加 / 减 / 乘 |
| divide | 两个 numeric | a / b；abs(b) ≤ 1e-12 时为缺失 |
| log1p_abs | numeric | log(1 + abs(x)) |
| sqrt_abs | numeric | sqrt(abs(x)) |
| square / abs | numeric | x² / abs(x) |
| clip | numeric | params 必须恰好含 lower、upper，且 lower < upper |
| is_missing | 任意类型 | 缺失为 1，否则 0 |
| frequency | categorical | 训练数据相对频率；未知类别为 0 |
| cross | 两个 categorical | 带长度前缀的分类交叉键 |
| group_mean | categorical, numeric | 按第一个字段分组的训练均值；未知类别用全训练均值 |
| group_std | categorical, numeric | 训练总体标准差 ddof=0；未知类别用全训练标准差 |
| year / month / dayofweek | datetime | 年 / 月(1–12) / 周几(周一=0) |
| days_between | 两个 datetime | (a−b) 的秒数 / 86400 |

除 clip 外 params 必须为空。无穷结果转为缺失；缺失补齐由后续 fold 内预处理负责。日期的 is_missing 在转为数值前计算。特征顺序决定编译顺序，不排序依赖。

## 主要对象

### create_study / FeatureStudy

| 参数 | 默认 | 说明 |
|---|---|---|
| metric | auc | 见指标表 |
| direction | 按 metric 推断 | maximize / minimize；显式方向与指标冲突时报错 |
| sampler | RandomSampler(42) | BaseFeatureSampler 实例 |
| storage | None | 父目录，None 为内存模式 |
| study_name | default | 单一目录名，不允许路径穿越 |
| search_strategy | greedy | independent / greedy / beam / autonomous |
| beam_width | 3 | 每代 Top-K 数量 |
| max_features | 32 | 新特征上限，最大 64 |
| param_space | {} | sklearn 参数名 → 非空枚举列表 |
| random_state | 42 | 默认 sampler/evaluator 的种子 |

`optimize(X, y, schema, estimator=None, n_trials=30, evaluator=None, groups=None, timeout=None, patience=None, callbacks=(), catch=(ValueError, RuntimeError), refit=True)` 返回 study 自身。

提供 evaluator 后无需再提供 estimator；evaluator 的主 metric 必须与 study 相同。默认捕获候选提议/评估的 ValueError、RuntimeError，记录 FAIL 后继续；LLMError 总是以失败试验记录。baseline 失败直接抛出。`catch=()` 可让模型的原始异常抛出，方便排查。持久化异常不会被当成模型失败吞掉。

| 访问器/方法 | 结果 |
|---|---|
| best_trial | 最优 COMPLETE Trial，可能为 baseline |
| best_features | list[FeatureSpec] |
| best_value | 最优内层 CV 均值 |
| baseline_trial | number=0 的 COMPLETE Trial |
| trials | 含失败/中断/重复 trial 的完整列表 |
| trials_dataframe() | 一行一个 trial，包含嵌套反馈 |
| feature_history() | 一行一个 trial-feature，包含 hypothesis 和证据 |
| concept_summary() | concept/status 的记录数 |
| stop() | 请求当前 trial 后停止 |
| export_pipeline(path=None) | 已 refit 的 sklearn Pipeline，可选原子写入 joblib |
| export_features(path) | 最优 Proposal 的 JSON |
| report(path=None) | 自包含 HTML，可选写文件；默认包含 Plotly |
| load_study(storage, study_name, sampler=None) | 从 SQLite 加载历史，训练数据仍需重新提供 |

`autonomous` 默认使用 `AutonomousSampler`：数值提议与可选 LLM 提议交替，定期尝试删除叶子特征；每轮最多新增一列，达到 `max_features` 后继续从当前最优集合尝试删除。新列若在每个训练折都恒定，记为 `SCREENED_OUT`，不消耗模型拟合。加入和删除都按相同 folds 的完整 CV 评价；默认连续 8 轮不改善即停止，baseline 始终可选。语义提议需显式配置 `AutonomousSampler(llm_sampler=LLMSampler(features_per_trial=1))`。`max_features` 是安全上限。

Trial 状态：RUNNING、COMPLETE、FAIL、DUPLICATE、SCREENED_OUT、INTERRUPTED、BUDGET_EXCEEDED。编号 0 是 baseline，之后连续递增，重启不重用中断编号。value/metrics 是原始方向的分数；delta=value−baseline，parent_delta=value−parent。对 minimize 任务，负 delta 表示提升。

### CVEvaluator

构造参数：`estimator=None, metric="auc", cv=5, metrics=None, random_state=42, n_jobs=1, importance_repeats=None, importance_max_samples=500`。设置 importance_repeats=0 可关闭数值解释以降低搜索成本，此时正向 delta 的假设缺少 importance 支持，会标为 inconclusive。

`prepare(X,y,groups)` 固定并检查 splits；`evaluate(X,y,schema,features,model_params=None)` 返回 EvaluationResult。直接使用时，在更换数据后先重新 prepare。Study 自动完成 prepare。

| metric | 优化方向 | 定义 |
|---|---|---|
| auc | maximize | 二分类 ROC AUC |
| auc_ovr | maximize | 多分类 weighted one-vs-rest AUC |
| accuracy | maximize | 准确率 |
| f1 | maximize | weighted F1 |
| log_loss | minimize | 交叉熵损失，返回正值 |
| rmse | minimize | 均方根误差，返回正值 |
| mae | minimize | 平均绝对误差，返回正值 |
| r2 | maximize | 决定系数 |

EvaluationResult 含 value、metrics、fold_values、importance、directions、duration。CV 均值对每个 fold 等权。每一折训练完全独立，候选失败不会污染其他候选。

### 假设反馈

- `retained`：父集合已有特征，在当前 trial 保留。
- `strongly_supported`：组合相对父集合改善、该列 permutation importance > 0、至少 80% folds 改善且预期方向未知或与观察方向一致。
- `supported`：组合改善、该列 importance > 0，且观察方向不与预期冲突，但未满足上一条全部条件。
- `rejected`：组合未改善。
- `inconclusive`：组合改善但缺少正 importance，或观察方向与预期冲突。

stability=high 表示至少 80% folds 的组合分数改善，否则 mixed。方向为数值特征与目标的验证集 Spearman 相关均值，绝对值不超过 0.05 时 unknown；非数值目标不输出方向。所有判断都是启发式反馈，不是显著性检验、个体特征消融、因果推断或 SHAP。

### LLMSampler / LLMClient

LLMSampler 参数：`model, temperature=None, client=None, history_limit=8, max_context_chars=24000, repair_attempts=1, features_per_trial=8, context_builder=None, context_budget=None, retriever=None, retrieval_threshold=200, **client_kwargs`。features_per_trial 为硬上限，1–32。可注入自定义 LLMClient（测试中使用 httpx MockTransport）。

LLMClient 参数：`model=None, provider="anthropic", base_url=None, api_key=None, temperature=None, max_tokens=2048, output_token_parameter=None, timeout=60, retries=2, token_budget=None, input_cost_per_million=None, output_cost_per_million=None, cost_budget=None, transport=None`。`temperature=None` 不发送该字段；OpenAI 默认使用 `max_completion_tokens`，Anthropic 和本地 OpenAI-compatible 默认使用 `max_tokens`。旧式 OpenAI-compatible 网关可指定 `output_token_parameter="max_tokens"`。

未显式配置时读取 FEATUNE_MODEL、FEATUNE_BASE_URL、FEATUNE_API_KEY；默认 model 为 claude-opus-4-8，默认 Anthropic 服务地址为 api.anthropic.com。使用其他供应商时显式设置其实际支持的 model。`usage` 包含 input_tokens、output_tokens、requests、unknown_requests；`total_tokens` 为已知输入与输出之和。

## CLI

```bash
featune optimize examples/config.json
featune inspect --storage runs --study-name cli-demo
featune report --storage runs --study-name cli-demo --output runs/cli-demo/report.html
featune export --storage runs --study-name cli-demo --output runs/cli-demo/best-features.json
```

配置文件见 [完整样例](../../examples/config.json)。路径相对于配置文件目录。data 支持 CSV 和 Parquet（Parquet 需要 pyarrow）；target 自动从输入列移除；可配置 groups 列。estimator 类型：tabpfn（默认）、tabpfn_classifier、tabpfn_regressor、logistic、ridge、hist_classifier、hist_regressor、random_forest、torch_classifier、torch_regressor。sampler 类型：random、evolution、llm、hybrid、autonomous；自主模式的 `llm` 子配置可选。

重复执行 optimize 命令会继续增加 n_trials；不是幂等查询。report/inspect/export 是只读操作，export 导出可审阅特征 JSON，拟合模型由 optimize 输出。使用 `featune --verbose optimize ...` 调整控制台日志。库通过 `logging.getLogger("featune")` 提供日志，应用可自行配置 handler。

## Feature IR、FeatureSet 与 lineage

`FeatureSet.from_features(schema, features, source_trial=None, origins=None)` 将受控特征规范转为唯一执行表示。FeatureIR 附加 output_dtype、stateful、lineage_parents（输入名→hash）、generation、source_trial、content_hash 和 version；hypothesis 保存语义概念。hash 包含操作、输出名、参数及依赖 hash，不包含理由与来源 trial。独立特征列表顺序不影响集合 hash；工厂执行拓扑排序与循环检测，原始 sampler 提议仍要求依赖顺序。

`FeatureSet.model_validate_json(feature_set.model_dump_json())` 可完整往返。compiler 会先把公开 FeatureSpec 输入正规化为 IR 再执行。`study.lineage_dataframe()` 返回跨 trial 的 DAG。trial 保存 parent_set、proposal、candidate_set、feature_count、parent_delta、fold_values、model_fits、transform_time、training_time。

## 累计预算、指纹与缓存

```python
budget = featune.SearchBudget(max_trials=20, max_model_fits=80,
                             max_wall_time=600, max_valid_proposals=15,
                             max_llm_tokens=100_000)
study = featune.create_study(budget=budget, joint_optimization=False,
                            include_statistics=False, cache=True)
```

所有上限均可选，并跨恢复累计；`n_trials` 表示本次新增尝试数。模型拟合次数包含 baseline folds、候选 folds 和最终全训练集 refit。时间预算在 folds/trials 间检查，不能中断已经运行的第三方 fit。预算耗尽正常停止并保存状态；baseline 未完成时没有 best_trial，refit 预算不足时没有可导出的 pipeline；已有缓存 refit 可以直接导出。`study.consumption` 返回累计用量。

`max_llm_cost` 必须同时提供 `input_cost_per_million` 和 `output_cost_per_million`；未提供价格时成本为未知 `None`，不会记为免费。所有价格须使用同一币种，属于估算而非供应商账单；缓存输入也按输入单价估算。请求前使用 UTF-8 字节数与最大输出量保守预留；存在未知计费请求时阻止继续发送预算受限请求。供应商不同缓存定价需另行对账。

`study.fingerprint` 记录库/源码/DSL/compiler/prompt 版本、Git revision、Python/依赖版本、dataset/schema/CV/estimator 指纹、sampler/provider/model 和 seed。指纹不兼容时拒绝恢复。显式 `optimize(..., allow_incompatible_resume=True)` 将旧 metadata/trials 归档到 SQLite，重置当前计数，启动独立实验，不混合旧结果。

六层缓存覆盖 IR、FeatureSet、fold transformer、转换矩阵、拟合 estimator 和评估结果；键包含实验命名空间及数据/schema/folds/FeatureSet/estimator/compiler 身份。磁盘缓存跨恢复保留，内存中的大型层各保留 8 项。`cache=False` 禁用缓存文件，但已完成的完全重复 trial 仍复用历史分数。缓存与导出 joblib 是可信本地工件，可能包含训练矩阵或统计，应保护目录且不得加载不可信 joblib；SQLite 与 LLM 提示不包含原始样本行。

`include_statistics=True` 可发送传入 optimize 的训练数据的 target-free 摘要：类型、缺失率、基数、数值统计/分位数以及不含类别名称的类别频率。默认关闭；outer test 绝不可传入 optimize。联合模型调参必须显式 `joint_optimization=True` 并设置允许的 param_space。

`search_summary(target_gain=0.005)` 返回候选评估数、模型拟合数、tokens/成本、有效/无效提议率、缓存命中率，以及达到最优、固定增益、最终增益 95% 所需资源；未达到的目标为 null。目标达成只依据内层 CV；minimize 指标按下降量计算增益。`pareto_frontier(resource="evaluations")` 还支持 model_fits、wall_time、tokens、api_cost；未知成本不能进入成本 Pareto。HTML 报告包含这些指标、特征集合演化和 lineage。

`study.proposal_history()` 以及报告中的“LLM / sampler proposals”表会展示解析后并经过校验的 proposal：operator、输入字段、hypothesis 的 concept/rationale、模型参数、状态、错误和 token 用量。每轮 INFO 日志同步输出通过校验的假设、评估状态/原因、parent_delta、缓存命中与最终选优决策；每折完成时输出主指标。原始模型响应不会持久化或写入日志；结构化 proposal 才是实际执行和审计的对象。

## 大字段与受控上下文

```python
from featune import ContextBudget, ContextBuilder, SemanticColumnRetriever

context = ContextBuilder(
    budget=ContextBudget(max_columns=40, max_detailed_stats_columns=20,
                         max_history_trials=10, max_failed_patterns=10,
                         max_tokens=12000, max_concepts=6, max_lineage_items=20),
    retriever=SemanticColumnRetriever(top_k=40, concept_top_k=5, seed=42),
    retrieval_threshold=200,
)
# 将 context_builder=context 传给 LLMSampler；构建对象不会调用 API。
```

FieldSchema 支持 `semantic_tags`（最多 20 个短标签）、`entity`、`encoding`、`available_at`；DatasetSchema 可选 `target_definition`、`prediction_point` 和 `search_guidance`（最多 2000 字符的不可信搜索假设）。后两者说明标签定义及预测时点，不发送逐行目标值。prompt 还包含实际 estimator 类名及可选 TabPFN 模型版本。字段可用性只是描述，事后字段仍须显式排除。概念依次采用首个标签、partition、entity、dtype；分区描述参与检索。默认本地语义索引使用字段名、描述、标签、实体、单位、编码、可用时点和类型的字符 TF-IDF，无远程 embedding 依赖，不保证识别任意同义词。可通过 BaseColumnRetriever 替换；另提供 RandomColumnRetriever、RuleBasedColumnRetriever 和 HybridColumnRetriever，后者保留一半名额做固定 seed 的随机探索。固定 schema、历史、seed 与 trial 编号时，选择可重复。

默认达到 `retrieval_threshold=200` 时启用检索；字段数超过 `max_columns` 时也提前启用。Retriever 先按任务与有效历史概念缩小范围，ContextBuilder 再约束字段／概念数并分配信息量，自定义 Retriever 也不能突破预算。描述压缩到 240 字符。当前 parent 名称与依赖保留，新提议不能引用上下文之外的原始字段。调高 threshold 不会绕过字段硬上限。

SearchMemoryCompressor 从真实内层 trial 产生有效／拒绝概念、最佳特征、成功／失败的组合交互、失败算子、无效提议阶段计数与近期前沿。“strong”仅表示相对 parent 的组合改善，不是统计显著性或因果结论。1000 条历史后的各类条目仍有上限。无效响应始终脱敏，不保存任意生成表达式作为失败模式。

`token_allocation` 默认 schema 0.40、memory 0.25、statistics 0.20、instructions 0.15。比例是软分配，必须保留的 task／DSL／parent 可使用空余额度；最终 prompt 则严格检查 UTF-8 字节数构成的保守 token 预留和字符上限，并提前预留所有 repair 后缀。该预估不等于真实计费 tokens。超限依次缩减详细统计、低优先级记忆、lineage 细节和低优先级字段；不能容纳必要约束时，在 API 调用前明确失败。

`include_statistics=True` 时只按需计算检索候选中最多 `max_detailed_stats_columns` 个字段的 target-free 统计，并在同次 optimize 内复用。Builder 不接收外层测试数据或分数；调用者必须只向 optimize 提供训练行，框架无法自动识别任意 DataFrame 是否属于测试集。统计不包含分类原始标签或样本行。

Trial 的 `context_metadata` 持久化最终字段、检索分数／原因、检索／构建时间、prompt hash 与 token 预留，不保存 prompt 或样本数据。Context／Retriever／Memory 版本和配置进入 ExperimentFingerprint；改变配置默认拒绝恢复，只能显式归档后开始新实验。CLI 的 LLM sampler 配置也支持 `context_budget={...}` 和 `retrieval_threshold`。

## 正确性与缓存兼容性修正（2026-09-22）

- 所有数值输入在运算前转换为浮点数，避免无符号减法、整数平方等静默回绕；非有限浮点结果仍转为缺失值。
- 同一 CVEvaluator 更换 X/y 后，在缓存读取前报错；需显式 `prepare(X, y, groups)`，FeatureStudy 已自动处理。原地修改数据也会被检测。
- `content_hash` 保留名称用于 lineage；`FeatureSet.expression_hash` 递归忽略展示别名，仅用于表达式比较。Trial 去重、evaluation cache、矩阵与拟合 pipeline 都区分列名，因为 DataFrame estimator 可能检查列名。
- Compiler version 3、prompt version 4、context version 2 会使旧实验指纹失效。旧历史可读，不能无校验混合继续搜索；可显式归档后重启。
- OpenAI-compatible 的空 choices 等异常响应记录为受控 LLM 失败，不再中断整个 Study。

## 默认 TabPFN 模型

Built with PriorLabs-TabPFN

`pip install featune` 一次安装 TabPFN、PyTorch 和 Plotly。`visualization` / `torch` extra 仅作为兼容旧安装命令的空别名保留。开发安装使用 `pip install -e '.[dev]'`。

默认固定 **tabpfn==9.0.0 与 TabPFN-2**，分类和回归各自的 revision 与 SHA-256 见 `featune/tabpfn.py`。显式选择 `model_version="v3.5"` 才使用标准 3.5 权重（`tabpfn-v3.5-20260909.safetensors`，revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`），分类/回归共用。包版本与模型代际独立，当前可选值为 `v2` 和 `v3.5`，没有隐式升级。v2 许可见 `LICENSES/TabPFN-v2.txt`，可选 3.5 许可见 `LICENSES/TabPFN-3.5.txt`。v2 分类权重经上游微调，公共数据集评估需考虑训练数据重合的可能。

省略 `optimize` 的 estimator/evaluator 或使用 `CVEvaluator()` 时，按主指标选择任务：auc、auc_ovr、accuracy、f1、log_loss 使用分类；rmse、mae、r2 使用回归。不会根据整数标签猜测任务。搜索方向按指标自动推断；显式方向与指标冲突时报错。显式 estimator/evaluator 仍优先，不会被默认模型替换。

```python
study = featune.create_study(metric="rmse", direction="minimize")
study.optimize(X, y, schema, n_trials=3)

# Explicit device/ensemble settings; no change to the fixed checkpoint.
evaluator = featune.CVEvaluator(
    featune.TabPFNRegressor(device="cpu", n_estimators=2, random_state=42),
    metric="rmse", cv=3,
)
```

`TabPFNClassifier` / `TabPFNRegressor` 参数为 `model_version="v2", n_estimators=4, device="auto", random_state=42`。集成次数与种子固定，不跟随上游自动配置。支持 sklearn clone/get_params/set_params 和联合搜索，例如 `param_space={"n_estimators": [1, 2, 4]}`；不支持用学习率或 epoch 调整该固定权重推理模型。

模型直接接收 fold 内编译的数值/类别 DataFrame，由 TabPFN 完成预处理；每个 fold 都创建独立训练上下文。默认 v2 上限为 10,000 训练行、500 列、分类 2–10 类，CPU 最多 1,000 行。可选 v3.5 的行/列/CPU 上限为 1,000,000/20,000/5,000，适配器仍限定分类 2–10 类；这些数字不保证当前硬件能承载最大规模。最终 refit 使用全部传入训练数据，也受这些限制。超限报错，不自动抽样、解除限制、换模型或调用云端服务。大数据可显式传入其他 sklearn estimator。

`importance_repeats=None` 在 TabPFN 下选择 0，在其他 estimator 下选择 2。开启数值解释时显式设置正整数，并保持 `n_jobs=1`，避免复制 GPU 状态；额外预测会增加耗时。关闭解释时，报告没有 permutation importance，依赖该证据的假设可能显示 inconclusive。Trial 新增 `prediction_time` 和 `explanation_time`，单位为秒；`model_fits` 表示 fit 调用次数，不等于梯度训练次数或总计算量。

默认 v2 首次拟合下载约 29 MB 分类权重或 44 MB 回归权重，无需 3.5 的浏览器授权流程。仅显式选择 v3.5 时，首次未缓存拟合才会先调用官方许可流程，再下载约 876 MB 权重。带桌面浏览器的交互式终端会自动打开登录页，等待用户登录、确认许可及浏览器回调，然后自动继续下载；Featune 不代用户接受条款。Notebook/非交互/无浏览器环境需先在 https://ux.priorlabs.ai 完成授权并配置 `TABPFN_TOKEN`；设置 `TABPFN_NO_BROWSER=1` 可在缺少凭证时快速报错。如 Hugging Face 要求仓库权限，另行配置 `HF_TOKEN`。不要提交凭证。安装依赖不下载权重，也不授予模型使用权。

权重解析不会上传训练数据。完成一次已授权的小样本 fit 后，设置 `HF_HUB_OFFLINE=1` 验证缓存运行；离线缓存缺失直接失败，不触发浏览器或认证请求。`HF_HOME` 控制缓存位置，仅 v3.5 分类和回归复用同一权重。显式配置的 LLM sampler 所需网络访问与本地 TabPFN 推理是独立的。

`export_pipeline` / joblib 缓存保存上游 fitted-state archive，不复制基础权重，不依赖源机器的绝对缓存路径。新环境需要相同包版本和固定权重缓存，首次预测会验证校验值并恢复状态；未缓存时需允许下载。`device="auto"` 在新机器按可用硬件恢复；显式指定的 GPU 不会自动改为 CPU，需要在首次预测前 `pipeline.set_params(model__device="cpu")`。跨硬件数值可能有小幅差异。

**TabPFN 模型工件包含训练特征和标签的上下文，不只是抽象参数。** joblib 与模型缓存应按训练数据保护，不能作为脱敏结果发布；仅加载可信工件。HTML 报告与 SQLite 搜索记录不因此包含原始数据行。发布下游模型或服务时，还应遵守下列独立许可约束。


### 许可与运行约束

Featune 代码继续采用 MIT，TabPFN 包代码独立采用 Apache-2.0。以下限制适用于**可选 v3.5**。默认 v2 采用独立的基于 Apache-2.0 的 Prior Labs 许可，并包含署名和下游模型命名要求，以随包原文为准。模型使用权由 Prior Labs 依据 [TabPFN-3.5 License v1.0](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/LICENSE) 直接授予（2026-09-09 修订），下述摘要不替代完整条款。

- 允许符合条款的非商业研究、测试和有限内部评估。商业或生产用途需要另行取得商业许可，即使 Featune 或你的应用是开源项目也不例外。
- 托管服务、API、SaaS，无论收费还是免费，均需另行获得商业许可。用于生产、业务流程、客户交付的模型输出同样受限；不得用输出训练、微调或蒸馏竞争模型。
- 许可排除德国 BGB 第 13 条定义的消费者，要求符合第 14 条的专业用途身份。再分发还涉及许可副本、署名及衍生修改声明要求；适用时附带上游许可与 `NOTICE`。具体授权请向 Prior Labs 确认。
- 容量数字是预设保护上限，不保证最大规模可在当前硬件运行。适配器明确支持分类 2–10 类，CV 与最终 refit 都需满足内存限制。固定权重推理没有学习率或 epoch 训练控制。
- 切换 v2/v3.5 会改变搜索来源指纹，请使用新 study 目录并重新拟合。对模型选择接口引入前的旧工件保留原环境，不得修改来源指纹来强制复用。
- 真实权重验收需要已授权的模型访问。普通测试使用受控替身，不能证明真实模型的数值质量或导出兼容性；发布前请执行显式开启的集成测试。


### 显式选择模型

```python
model = featune.TabPFNClassifier(model_version="v3.5")
study.optimize(X, y, schema, estimator=model, n_trials=3)
```

CLI JSON：

```json
{"estimator": {"type": "tabpfn", "params": {"model_version": "v3.5"}}}
```

省略 `model_version` 即使用 v2；回归器 `TabPFNRegressor` 使用相同参数。
