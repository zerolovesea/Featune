# Featune 设计与实施状态 / Design and implementation status

Created: 2026-09-21 · Updated: 2026-09-23 · Package: 1.0.0

本文件按当前代码整理，是开发交接入口；已实现能力与待办分开描述。代码契约以源码和自动测试为准，API 细节见[中文参考](docs/zh/reference.md) / [English reference](docs/en/reference.md)，实验结论见[基准文档](benchmarks/README.md)。不再将历史计划中的建议 API 或未完成实验写成已交付能力。

This is the current implementation map, not a promise that every original research goal has been met. Source and tests define behavior; benchmark summaries define the limits of empirical claims.

## 1. 定位与当前交付 / Scope and delivery

Featune 是受控特征搜索库。LLM 是可替换的提议器，不能执行代码、改写训练循环或直接决定最终模型。所有采样器均走相同的结构化校验、编译、内层 CV 和预算管理路径。

| 能力 | 当前状态 | 主要实现与测试 |
|---|---|---|
| Schema、任务/标签/预测时点语义、封闭 DSL、Feature IR / FeatureSet、来源 DAG | 已实现 | `schema.py`, `ir.py`; `test_core.py`, `test_search_contracts.py` |
| fold 内编译、配对 CV、多指标、分组/时间切分 | 已实现 | `compiler.py`, `evaluation.py`; `test_core.py`, `test_search_contracts.py` |
| Random / Evolution / LLM / Hybrid / Autonomous | 已实现；自主搜索收益未证实 | `samplers.py`, `study.py`; `test_llm.py`, `test_autonomous.py` |
| independent / greedy / beam / autonomous、预算、去重、恢复 | 已实现 | `study.py`, `budget.py`, `storage.py`, `cache.py`; `test_search_contracts.py` |
| 有界字段检索、按需统计、历史压缩 | 已实现；真实语义收益未证实 | `retrieval.py`, `context.py`, `memory.py`; `test_context.py` |
| 默认固定 TabPFN v2 | 已实现并完成真实权重验收 | `tabpfn.py`; `test_tabpfn.py` |
| 显式选择 TabPFN v3.5 | 授权/来源/格式路径有模拟测试；未完成真实权重端到端验收 | `tabpfn.py`; `test_tabpfn.py` |
| PyTorch 表格模型、trial 间有限参数搜索 | 已实现；不支持 epoch 中途改模型并续训 | `torch.py`; `test_torch.py` |
| SQLite、逐轮假设/判断日志、导出、HTML/Notebook、CLI | 已实现 | `storage.py`, `study.py`, `reporting.py`, `cli.py`; `test_context.py`, `test_interfaces.py` |
| 六数据集 v2 对比 | 240/240 次运行完成；intermediate evidence | `benchmarks/run.py`, `summarize_tabpfn.py`; `test_benchmarks.py` |

默认安装已包含 `tabpfn==9.0.0`、PyTorch、Plotly、Hugging Face Hub，以及 numpy/pandas/sklearn/pydantic/httpx/joblib。`visualization` 和 `torch` extras 仅保留兼容名称。`benchmark` extra 包含 OpenFE 0.0.12、LightGBM、Matplotlib、Excel 读取依赖，并将 sklearn 限于 `<1.6`；使用独立环境。Python 最低版本为 3.10。

Default installation includes model and visualization dependencies. Foundation checkpoints are downloaded lazily, not shipped in the wheel. Benchmark-only dependencies remain separate.

## 2. 模块与数据流 / Architecture

```text
DatasetSchema + X/y（仅外层训练数据）
  → CVEvaluator.prepare：固定 folds 与数据身份
  → baseline Trial 0
  → parent FeatureSet → Sampler → Proposal
  → 校验算子、类型、依赖、参数范围 → FeatureIR / FeatureSet
  → 名称敏感的试验去重 / 兼容缓存；递归表达式身份用于比较
  → 每个 fold：FeatureCompiler.fit(train) → transform(train/valid) → 模型拟合/评分
  → Trial：分数、parent delta、成本、来源、解释、状态
  → SQLite / 搜索历史 / 下一轮 parent
  → 最优 FeatureSet 在全部输入训练数据上 refit
  → 导出 pipeline → 独立外层测试评分
```

`FeatureStudy` 编排流程，`CVEvaluator` 负责配对评估，`FeatureCompiler` 负责确定性受控变换。`LLMClient` 处理传输和计费，`LLMSampler` 处理提议及有限修复，`ContextBuilder` 决定提供多少信息，Retriever 决定选择哪些字段。不要在 CLI、报告或新增 sampler 中复制这些职责。

Study owns orchestration; evaluation and compilation are shared by all samplers. The caller owns the outer holdout and must not pass it into optimization or prompt construction.

## 3. 输入、输出与约束 / Contracts

- 输入是列名唯一的 DataFrame、对齐的一维无缺失目标和 `DatasetSchema`。Series 的 index 必须对齐；分组标签也需对齐。
- Schema 描述 dtype、语义、单位、编码、预测时点、标签定义及排除规则。目标/泄漏字段必须排除；`available_at` 是描述，时间可用性不能由库自动证明。
- Proposal 只能使用封闭算子和用户允许的参数选择。拒绝未知字段、非法类型/参数、同名冲突和循环/前向依赖；不执行生成的 Python、SQL、shell 或 lambda。
- `FeatureSet` 的规范身份不依赖输入排列；递归表达式身份支持跨别名比较，但评分与试验去重保留列名，因为 DataFrame estimator 可能依赖列名。
- 有状态 frequency/group mean/group std 只在训练 fold 学习，验证集使用已学映射和回退值。算术先提升浮点类型以避免整数溢出；分类交叉采用无歧义编码。
- 特征价值通过整个 candidate FeatureSet 的 CV 评价。搜索方向按主指标推断，冲突的显式方向被拒绝。`delta` 为 candidate 减 baseline，`parent_delta` 为 candidate 减 parent；对于 RMSE，负差值才是改善。
- 输出包含 `best_trial`, `best_features`, `best_value`, `trials_dataframe()`, `feature_history()`, `lineage_dataframe()`, `search_summary()`, `pareto_frontier()`、特征配置与拟合后的 pipeline。

The DSL is an execution boundary, not a parser for arbitrary model-written programs. Scores and differences retain their metric's direction; explanatory text is not executable evidence.

## 4. 默认模型与持久化 / Model presets and artifacts

默认 `TabPFNClassifier` / `TabPFNRegressor` 使用 `model_version="v2"`, `n_estimators=4`, `device="auto"`, `random_state=42`。CVEvaluator 在未传 estimator 时按指标选分类器或回归器。TabPFN 接收保留缺失/类别信息的 DataFrame；普通 sklearn 模型仍走相应 fold-local 预处理。失败不会静默回退到另一种模型。

| 预设 | 训练行上限 | 列上限 | 分类类别 | CPU 训练行上限 | 状态 |
|---|---:|---:|---|---:|---|
| `v2` | 10,000 | 500 | 2–10 | 1,000 | 默认；真实权重分类/回归/离线导出验收通过 |
| `v3.5` | 1,000,000 | 20,000 | 2–10 | 5,000 | 用户显式选择；实际还受内存和设备约束 |

库固定 TabPFN 包版本与 checkpoint revision/SHA-256，具体值见 [tabpfn.py](featune/tabpfn.py)。安装依赖不等于已下载权重，也不等于用户取得可选模型的授权。

- 优先使用本地缓存，校验权重摘要；离线缺失直接报错，不发起授权。
- v3.5 未缓存时先调用官方许可流程。可交互终端打开登录页，用户完成授权后通过官方回调继续下载；无头/不可交互环境必须预配置凭证，浏览器登录不会自动给任意 CI 授权。
- safetensors 缓存通过保留文件扩展名的硬链接处理上游解析路径问题。
- fitted state 保存训练上下文和固定模型来源，不打包基础 checkpoint；加载时恢复当前缓存路径并校验预设。基础权重仍需在目标机器具备合法访问权限。
- SQLite/cache/pipeline 都是可信本地工件；joblib 不能加载不可信文件。模型导出可能含训练数据上下文，不能当作脱敏结果公开。
- 项目代码 MIT 与上游权重许可分开；使用约束、v2 署名要求和 v3.5 限制见 [NOTICE](NOTICE) 及 [LICENSES](LICENSES)。不把项目开源许可解释为上游权重的无限授权。

V2 is the tested default. V3.5 is explicit opt-in, with separate authorization and licensing. Passing mocked authorization tests does not establish successful real-weight inference.

## 5. 搜索、预算、缓存与恢复 / Search lifecycle

当前默认 greedy、beam width 3、最多 32 个派生特征、`optimize(n_trials=30)`；自主模式需显式选择 `search_strategy="autonomous"`，默认数值提议，可配置单列 LLM 提议；Random 每次提议 1 列，LLMSampler 默认最多 8 列。基准的“四次尝试、每次一列”是实验配置，不是库默认值。联合 HPO 默认关闭，开启时只允许用户声明的有限 `param_space`。

自主模式每轮只新增一列或删除父集合的叶子列；达到 `max_features` 后继续从当前最优集合尝试删除。新列若在每个训练折都恒定，先标记 `SCREENED_OUT`，不进行模型拟合。其余候选以同一组 folds 完整 CV 比较，默认连续 8 轮不改善时停止；`max_features` 仅是上限。LLM 与数值提议的交替由 `AutonomousSampler` 负责，未配置 LLM 时只做数值搜索。

Trial 状态包括 RUNNING、COMPLETE、DUPLICATE、SCREENED_OUT、FAIL、BUDGET_EXCEEDED、INTERRUPTED。baseline 为 Trial 0，始终可被选中；重复尝试不重新拟合，但保留试验记录。回调在 trial 边界执行，不是 epoch 训练回调。

`SearchBudget` 统一限制 trials、model fits、wall time、LLM tokens/cost、valid proposals。模型拟合前和请求前做预算检查；失败、修复及未知计费均影响累计资源。墙钟限制不能硬中断正在执行的第三方 fit/HTTP。最终 refit 也消耗预算，预算不足时不会导出未拟合占位模型。

SQLite 事务记录 trial、配置、RNG、预算与停止原因；同一 study 的并发写入被拒绝。六个缓存层为 IR、FeatureSet、fold transform、matrix、estimator、evaluation，按实验身份隔离；重型内存层最多保留八项。无 storage 时可使用内存缓存。

`ExperimentFingerprint` 覆盖数据/schema/CV/estimator、依赖、源码、DSL/compiler/prompt、sampler/context/retrieval/memory 配置等。当前 DSL=1、compiler=3、prompt=4、context=2、retrieval=2。源码 docstring 修改也可能改变源码指纹。不兼容恢复默认拒绝；`allow_incompatible_resume=True` 是显式归档/重启，不是强行复用旧缓存或把旧试验继续算成新实验。

Budgets are cumulative and checked at operation boundaries. Resume validates identity; an incompatible override restarts archived history rather than silently combining incompatible evidence.

## 6. LLM 与大字段上下文 / LLM context

支持 OpenAI-compatible 和 Anthropic Messages-compatible 协议，本地服务复用兼容接口。凭证通过环境变量配置；不得进入提示、日志或版本库。LLM 上下文包含任务、标签定义、预测时点、实际 estimator 类型与可选模型版本、选中字段的单位/编码/可用时点。默认不接收样本行和逐行目标值；`include_statistics=False`，显式开启时只计算选中训练字段的无目标统计，并过滤允许的统计键。

默认本地字符 TF-IDF 检索，另有 Random、RuleBased、Hybrid；不是远程 embedding 模型。`retrieval_threshold=200`，但字段数超过 `max_columns` 时同样启用检索，不能将“200 以下”理解为无限制完整上下文。

| ContextBudget | 默认 |
|---|---:|
| max_columns / max_detailed_stats_columns | 40 / 20 |
| max_history_trials / max_failed_patterns | 10 / 10 |
| max_tokens | 12,000 |
| max_concepts / max_lineage_items | 6 / 20 |
| schema / memory / statistics / instructions 分配 | 40% / 25% / 20% / 15% |

`max_tokens` 使用 UTF-8 字节数做保守预留，不是供应商 tokenizer 或真实账单；还受 sampler 的字符上限约束。SearchMemory 从真实内层试验压缩概念、失败模式和历史，ContextBuilder 按需统计并裁剪；任务和 DSL 约束不能被直接截断。当前是一条本地检索/压缩后提议路径，并未实现三个独立 LLM 请求的概念→字段→特征选择协议。

LLM 返回的 rationale 只是待核查假设。模型类型、编码含义、单位及因果叙述可能被误解；数值评估不自动认证叙述正确。未知计费请求和未配置单价分别记录，不将未知成本记成免费。

Retrieval limits prompt size; it has not been shown to improve real-task semantic search consistently. No row-level or outer-test evidence should be sent to the model.

## 7. 评估、解释与 Torch 边界 / Evaluation and training boundaries

整数 CV 对分类用分层切分、回归用 KFold，传 groups 时采用 GroupKFold；`cv="time"` 使用 TimeSeriesSplit，调用方须先按时间排序。自定义 folds 检查索引重叠和 group 穿越，baseline 与候选共享准备好的切分。

TabPFN 默认 `importance_repeats=0`，其他 estimator 默认 2；TabPFN 开启置换解释时要求 `n_jobs=1`。解释保存数值置换、方向关联、fold 稳定性及受限的假设状态。未计算相应证据时不能将 inconclusive 改写为 supported；既不是因果识别，也不是 SHAP。

自带 TorchClassifier/TorchRegressor 提供数值分支、类别 embedding、MLP 和 AdamW。每次 `fit` 重新初始化；宽度、学习率、embedding 等可在不同 trial/fold 的独立拟合之间搜索。**尚无 epoch 中途 LLM 调整学习率、切换优化器、修改网络结构、迁移优化器状态并接续训练的接口。** TabPFN 默认拟合也不开放这类训练循环控制。

训练期 INFO 日志在每轮提议通过校验后记录假设概念、理由、算子、输入和预期方向；每折结束记录主指标；每轮结束记录新假设的评估状态、依据、parent delta、缓存命中及候选选优处理。未经校验的 LLM 原始响应不写日志，失败只记录脱敏错误。HTML/Notebook 报告展示轨迹、成本、来源、假设、目标达成与 Pareto；CLI 提供 optimize/report/export/inspect。已有双语示例和配置样例属于受维护输入，不应当作测试垃圾删除。

## 8. 验证、实验与产物策略 / Validation and retention

最近完整功能验收：2026-09-23，93 项测试通过，包含 v2 真实权重及方向、列名缓存、分类编码、标签/模型 prompt 回归；Ruff 规则检查通过。测试源码继续保留。历史 Python 3.10 模型专项与安装构建通过记录不替代当前环境重新验收；v3.5 完整真实推理不在本轮验收范围，真实 LLM 大字段消融已有单 seed 初步结果，未完成多 seed 验收。

此前 v2 协议：六数据集 × 五 seed，最多 1,200 行；120 次基础对照 + 60 次语义对照 + 60 次传统模型基线，全部完成。8 次 LLM 提议被预算保护阻止；43 次 Evolution 重复提议复用结果。语义 LLM 的六个任务改善区间均包含零，不能宣称普遍领先。该汇总产生于本次 prompt/compiler 版本更新之前，不能直接声称反映当前实现；详见[结果报告](benchmarks/results/tabpfn-v2/report.md)。

仓库只保留 Markdown 设计/协议/结论/对比表及报告引用的图表。逐次 JSON/CSV、trial/lineage/fingerprint、prompt 调试、SQLite/cache、HTML 运行报告、下载数据、模型工件放入被忽略的 `runs/`，不随源码发布。清理后保留的是汇总资料，不能据此重算所有逐次统计；如需审计和重新汇总，应在本地重跑并保留该次完整工件。不得声称已删除的原始审计记录仍在仓库。

旧 Retail 全量目标截尾实验已撤回，不用于准确率结论。历史有效比较只保留汇总并标注规模/模型/环境，不能与新 v2 子样本混合排名。

Only conclusions, aggregate comparison tables, protocol/environment notes and referenced figures are retained in Git. Raw runs remain a local reproducibility concern; regenerate them under ignored `runs/` when needed.

## 9. 本轮实现与剩余验收 / Delivery and remaining validation

| 顺序 | 已完成 | 尚需验证 |
|---|---|---|
| 1 自主特征集合搜索 | 单列语义/数值提议、训练折低成本筛选、完整 CV 增删、baseline 回退、patience/预算停止；120 字段 smoke 通过 | 更多真实宽表任务与明确收益；目前没有证明自主搜索优于 Random 或 baseline |
| 2 真实 LLM 语义审核 | 三任务模型/标签声明审核；当前上下文三例正确；发现时点依据被过度表述，已收紧提示词 | 更多任务与模型的误述率；描述性 `available_at` 仍不能证明无泄漏 |
| 3 独立新协议 | 新的 Wine/Digits 外层 holdout、同 trial 数/特征上限/拟合上限，Digits 4/8 次尝试预算曲线；保留负面结果 | 等实际拟合次数的精确匹配、大样本多 seed 与统计区间 |
| 4 真实 LLM wide-context | 200/500/1000 字段、五策略、单 seed 的真实请求与 token/外层分数记录；500/1000 的 full 策略受预算保护 | 多 seed 与真实宽表任务；不能从单次合成实验推断语义收益 |
| 5 默认 v2 复验 | 按用户范围只复验 v2，真实权重分类/回归/恢复测试 12 项通过 | v3.5 不在本轮范围，保留显式可选且未经真实权重验收的说明 |

自主搜索的高阶特征每轮只加一列，所以候选增益相对包含其低阶输入的父集合评价。低成本筛选只用训练折的无标签列值；未实现有标签预筛。加入和删除的来源、理由、筛选状态与停止原因进入 trial、日志和报告。新实验汇总见[自主搜索与上下文检查](benchmarks/results/autonomous-20260923.md)。原始记录仅保留在忽略的 `runs/`。

后续修正：自主搜索删除叶子列时跳过同一父集合已尝试的删除候选，满额父集合无可用删除时改选其他可扩展父集合。LLM 请求默认省略 temperature；OpenAI Chat Completions 默认使用 `max_completion_tokens`，兼容旧网关可指定 `max_tokens`。中英文 notebook 已重排为从数据到报告的完整教程，LLM 单元格为可选步骤；真实运行验收以本次执行记录为准，不从模拟协议测试推断任意网关兼容性。

新增 DSL 算子必须同步类型签名、校验、编译、隔离测试和双语参考；模型/缓存/提示协议变化同步版本与指纹。每次交付同时更新本文件、相关指南、测试状态及产物保留范围。
