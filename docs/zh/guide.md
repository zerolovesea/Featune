# 使用指南

[English](../en/guide.md) · [首页](../../README.md) · [API](reference.md)

## 数据与语义

如果是第一次使用，先运行 [五分钟示例](../../README.md#五分钟示例)；完整的大数据训练与竞赛提交见 [Playground notebooks](../../examples/kaggle/README.md)。Playground 中数十万行数据超过默认 TabPFN v2 的容量，示例显式传入 LightGBM，并在特征搜索结束后用全部训练行重新拟合获选 pipeline。

输入 `X` 为列名唯一的 pandas DataFrame，`y` 为同长度数组或索引与 X 一致的 Series。目标不能放进可用字段中。schema 可使用 Pydantic 对象或对应字典：

```python
from featune import DatasetSchema, FieldSchema
schema = DatasetSchema(
    fields=[
        FieldSchema(name="loan", description="申请时贷款本金", dtype="numeric", unit="CNY", partition="finance"),
        FieldSchema(name="income", description="申请时月收入", dtype="numeric", unit="CNY/month", partition="finance"),
        FieldSchema(name="region", description="申请时所属地区", dtype="categorical", partition="profile",
                    encoding="行政区划代码；缺失表示未填写", available_at="申请提交时"),
        FieldSchema(name="settled_at", description="事后结清时间", dtype="datetime", exclude=True),
    ],
    partitions={"finance": "贷款决策前财务字段", "profile": "申请人资料"},
    objective="仅使用申请时可见信息预测未来逾期",
    target_definition="标签 1 表示申请后 90 天内至少一次逾期；0 表示没有逾期",
    prediction_point="申请提交时，放款及还款行为发生之前",
)
```

`exclude=True` 的字段不进入模型，也不能被特征引用。schema 之外的 X 列不进入模型。分区用于向 LLM 解释字段所属业务域，不是 train/test 分区。时间切分需要单独配置 CV。`objective`、`target_definition`、`prediction_point`、字段 `encoding` 和 `available_at` 会进入 LLM prompt，但不发送逐行标签或样本；这些说明不会自动识别未来泄漏，事后字段仍须显式排除。字段描述在 prompt 中最多保留前 240 个字符，重要编码约束宜写入 `encoding`。

数值列的 NaN/inf 在特征计算后由训练 fold 中位数补齐；空数值列保留。分类值使用包含 Python 类型的字符串编码，区分缺失及不同类型的同形值；sklearn 路径使用最多 128 类的一热编码和未知类忽略。日期统一 UTC；原始日期映射到 Unix epoch 起算的天数。已有 dtype 不符会在训练之前报错。

## 选择采样器

```python
import featune
random_sampler = featune.RandomSampler(seed=42, features_per_trial=2)
evolution_sampler = featune.EvolutionSampler(seed=42, features_per_trial=2)
llm_sampler = featune.LLMSampler(provider="anthropic", model="claude-opus-4-8", features_per_trial=2)
hybrid_sampler = featune.HybridSampler(llm_sampler=llm_sampler, random_sampler=evolution_sampler, llm_every=3)
```

- Random：按 trial 编号派生可重复 RNG，生成满足类型约束的表达式，也可使用父特征构造高阶特征。
- Evolution：随机变异加优秀历史特征重组；重组仍须满足当前父集合的依赖。
- LLM：读取字段/分区语义、目标、父集合、历史指标、importance 和失败记录。默认压缩为最多 8 条近期记录及受限最佳特征／概念摘要，再按 token 和字符预算裁剪。
- Hybrid：trial 1、4、7… 调用 LLM，其余调用指定的随机/进化采样器。

自定义采样器继承 `BaseFeatureSampler`，实现 `sample(context) -> Proposal`；有状态采样器还需实现 `state_dict/load_state_dict`，把影响提议的配置纳入 `configuration()`，以保证恢复校验有效。

## 独立评估

```python
from featune import CVEvaluator

evaluator = CVEvaluator(
    metric="auc", cv=5,
    metrics=["accuracy", "log_loss"], random_state=42,
    importance_repeats=2, importance_max_samples=500, n_jobs=1,
)
```

分类默认 StratifiedKFold，回归默认 KFold。传入 `groups` 后整数 cv 自动使用 GroupKFold。支持 `TimeSeriesSplit` 或 sklearn 自定义 splitter，也支持显式 `(train_indexes, validation_indexes)` 列表。时间序列必须先按时间排序；根据标签可用期设置 `gap`。有群组的数据必须让训练和验证群组隔离；框架会检查重叠。

```python
from sklearn.model_selection import TimeSeriesSplit
# ordered_X/ordered_y must already be sorted by event time.
time_evaluator = CVEvaluator(cv=TimeSeriesSplit(n_splits=4, gap=7))
```

每个 trial 克隆 estimator；编译器、统计映射、补齐、编码、标准化都只在该 fold 的训练数据上拟合。baseline 和所有候选使用相同 folds。importance 在验证集对编译后的特征逐列置换，使用 scorer 的“越大越好”方向，因此正值表示对预测有帮助。原始字段与其派生字段是分别置换的，不是连带重算后的总效应。

对超过默认 TabPFN 容量的大表，显式指定 sklearn 兼容模型；下面的 `X_search/y_search` 是预先留出独立验证集后的训练部分。搜索结束后，先用导出的已拟合 pipeline 检查独立验证集，再克隆同一结构并在全部官方训练行上重新拟合：

```python
from lightgbm import LGBMClassifier
from sklearn.base import clone

evaluator = featune.CVEvaluator(
    estimator=LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
    metric="auc", cv=2, importance_repeats=0,
)
study = featune.create_study(metric="auc", sampler=featune.RandomSampler(seed=42))
study.optimize(X_search, y_search, schema=schema, evaluator=evaluator, n_trials=2)
search_pipeline = study.export_pipeline()  # 已在 X_search 上拟合
final_pipeline = clone(search_pipeline).fit(X_all, y_all)  # 全部训练行
```

`importance_repeats=0` 节省计算，但不产生 permutation importance 证据。`LightGBM` 是示例的额外依赖，不属于 Featune 默认安装。完整数据准备、验证和提交见 [Kaggle 示例](../../examples/kaggle/README.md)。

## 特征集合搜索与预算

```python
study = featune.create_study(
    metric="auc", direction="maximize", sampler=random_sampler,
    search_strategy="beam", beam_width=3, max_features=16,
    storage="runs", study_name="credit-v1",
    joint_optimization=True, param_space={"C": [0.1, 1.0, 10.0]},
)
# X/y/schema must describe the same training dataset.
study.optimize(X, y, schema=schema, evaluator=evaluator, n_trials=30, timeout=3600, patience=10)
```

`independent` 总从 baseline 扩展；`greedy` 从当前最优可扩展集合扩展；`beam` 每一代固定 Top-K 前沿、轮流扩展，下一代重新选前沿。baseline 也参与保留，避免比原始模型更差时仍强行选新特征。每一轮可提议多个相互依赖的新特征，按列表顺序编译。

`n_trials` 是本次新增尝试次数，不含 baseline；失败或重复也消耗尝试次数，避免无限补样。`max_features` 只统计新特征。`patience` 是本次调用内连续未改善的尝试数。`timeout` 包含 baseline 时间，在 trial 开始前检查，不硬中断正在执行的训练。最终 refit 不受该 trial 调度截止时间限制。

模型参数只能取 `param_space` 的有限枚举值。模型、网络和特征可在同一 trial 一起改变；该情况下 delta 是联合变化的结果，不能单独归因于某个特征。分类 embedding 的宽度、隐藏层列表也可作为参数值。

```python
# A callback runs after persistence. Returning a value does not stop search; call stop().
def on_trial(study, trial):
    print(trial.number, trial.state, trial.value, trial.tokens)
    if trial.value is not None and trial.value >= 0.99:
        study.stop()
```

`optimize(..., callbacks=[on_trial])` 在当前 trial 结束后调用。`KeyboardInterrupt` 不被吞掉；已经完成的 trial 已提交到 SQLite。

## 断点恢复

```python
sampler = featune.RandomSampler(seed=42, features_per_trial=2)
study = featune.load_study("runs", "credit-v1", sampler=sampler)
# Use exactly the original schema, data, estimator and evaluator settings.
study.optimize(X, y, schema=schema, evaluator=evaluator, n_trials=20)
```

恢复必须提供相同采样器配置（包括 seed、每轮特征数）、数据、schema、模型、CV 和 importance 设置。密钥可更新；预算可增加。`load_study` 无采样器参数适合只读 inspect/report；继续非默认采样器搜索必须显式重建采样器。

项目文件位于 `runs/<study_name>/`：`study.sqlite3` 保存配置、sampler 状态、trial；`search.log` 实时记录每折指标、每轮通过校验的假设（概念、理由、算子、输入、预期方向）、评估后的状态与判断依据，以及该轮是否成为最优。CLI 默认在终端输出同样的 INFO 日志；库调用者可配置 `logging.getLogger("featune")`。无效 LLM 原始响应不写入日志，只记录脱敏错误。`.writer.lock` 防止并发写。CLI 还生成 `pipeline.joblib`、`features.json`、`trials.json`，可选 `report.html`。原始 X/y 不存入历史；日志可能含业务字段语义和 LLM 理由，应按相应数据策略保存。

进程意外结束遗留的 RUNNING trial 标记 INTERRUPTED，保留编号，不重放可能已计费的请求。请求 token 收据不确定时记录 unknown_requests；设置 token_budget 的 LLM 会停止后续调用。可以检查供应商账单后选择新的预算/研究项目；不要把未知消费当作 0。

## LLM 协议与预算

`LLMClient` 支持 Anthropic Messages、OpenAI Chat Completions 和本地 OpenAI-compatible。请求只发送文本 JSON 提议任务，不上传数据行。默认不发送 temperature；显式设置后由服务决定是否接受。OpenAI 使用 `max_completion_tokens`，旧式兼容网关可设 `output_token_parameter="max_tokens"`。

```python
# 官方 OpenAI Chat Completions：model 需填写账户可用且支持该接口的模型。
openai_sampler = featune.LLMSampler(provider="openai", model="your-chat-model")
# 自定义 OpenAI-compatible 网关：base_url 可为 API 前缀或完整 /chat/completions。
gateway_sampler = featune.LLMSampler(provider="openai", model="your-model",
                                    base_url="https://gateway.example/v1",
                                    output_token_parameter="max_tokens")
# Claude Messages-compatible：base_url 可为 API 前缀或完整 /v1/messages。
claude_sampler = featune.LLMSampler(provider="anthropic", model="your-claude-model",
                                   base_url="https://gateway.example")
```

三个示例均从 `FEATUNE_API_KEY` 读取密钥；自定义服务必须返回对应协议的响应与 usage。OpenAI Responses API 和网关专用字段不属于此兼容接口。

每次请求记录 input/output tokens，包含有效 JSON 修复尝试。默认 `repair_attempts=1`；最多初次加一次修复。拒绝未知键、未知操作/字段、错误类型、重复 JSON 键、非有限值、前向依赖和预算外参数。只接受 JSON 或单一 `json` fenced block，不从任意回复文本中猜测抽取对象。

默认请求超时 60 秒，429/部分服务端错误最多重试 2 次；传输中断不自动重发，避免不明计费。HTTP 错误只输出状态码，不持久化响应正文、密钥或原始无效输出。token budget 在调用前按 UTF-8 字节数加 max_tokens 做保守预留，实际消费以服务端 usage 为准。该预留不是 tokenizer 精确计数或供应商计费硬上限；服务端缺少 usage 时消费标记未知。

`client.cost` 根据用户提供的 `input_cost_per_million/output_cost_per_million` 估算，不查询实时价格；缓存 token 也计入输入总数，因此缓存折扣需要自行核对。服务无 usage、网络超时或意外进程退出可能导致实际账单高于已知 token 总数。

本地 HTTP 只接受 loopback 地址；远端要求 HTTPS；不跟随重定向。API key 使用 `FEATUNE_API_KEY`，不要写入 CLI JSON。字段 description 本身也可能包含敏感内容，需按实际服务的数据策略选择上传范围。

## PyTorch

```python
from featune.torch import TorchClassifier
network = TorchClassifier(hidden_dims=(64, 32), embedding_dim=8, learning_rate=0.001,
                          epochs=30, batch_size=128, device="cpu", random_state=42)
network_study = featune.create_study(
    sampler=featune.RandomSampler(seed=42),
    joint_optimization=True, param_space={"hidden_dims": [[32], [64, 32]], "learning_rate": [0.001, 0.003],
                 "embedding_dim": [4, 8]},
)
network_study.optimize(X, y, schema=schema, estimator=network, n_trials=10)
```

PyTorch 路径接收编译后的 DataFrame，不再一热编码。训练 fold 学习类别词典、embedding 和数值标准化，未知类别映射到 0。回归使用 `TorchRegressor`，自动在训练数据上标准化目标并在预测时还原。网络、学习率、dropout、weight_decay、epochs、batch_size 可调。支持 PyTorch 的 CPU/CUDA 等设备；自动化验收覆盖 CPU，GPU 数值确定性依赖 PyTorch/设备设置。自定义 sklearn-compatible 神经模型也可直接传入 evaluator。

## 解释与导出

```python
study.best_trial
study.best_features
study.best_value
study.trials_dataframe()
study.feature_history()
study.concept_summary()
study.report("runs/report.html")
study.export_pipeline("runs/model.joblib")
study.export_features("runs/features.json")
```

报告 HTML 自包含 Plotly，无需联网；notebook 使用 `IPython.display.HTML(study.report())`。概念汇总按 concept/status 统计试验记录，因此相同特征跨 trial 会重复计数，不是独立科学证据次数。

`export_pipeline()` 返回已在本次 optimize 输入的全部训练数据上 refit 的 Pipeline；新数据 `.predict/.predict_proba` 会自动应用完全相同的特征逻辑。设 `refit=False` 时不生成可导出的拟合 pipeline。只读加载历史不会重建训练数据，需要相同配置 `optimize(..., n_trials=0)` 才能重新 refit。`features.json` 是可审阅的 DSL，不包含模型权重。

仅对可信来源的 joblib artifact 调用 `joblib.load`，因为 Python pickle 格式可执行代码；推理环境应与训练依赖版本相同。

## 大字段搜索

通过 `semantic_tags`、字段描述和任务目标提供检索依据。`LLMSampler(context_budget={"max_columns": 40, "max_tokens": 12000})` 自动检索受限字段子集；Study 设置 `include_statistics=True` 可按需补充训练集统计。用 `study.trials[-1].context_metadata` 检查选择原因和 prompt 预算。详见 [上下文 API 与限制](reference.md#大字段与受控上下文)。

离线检索验收：`python benchmarks/wide.py --output runs/wide-smoke`，不消耗 API tokens，也不证明预测收益。真实 LLM 评估需显式添加 `--llm`。


Built with PriorLabs-TabPFN

默认使用固定 TabPFN-2；安装、容量、权重缓存和导出数据保护规则见 [参考文档](reference.md#默认-tabpfn-模型)。TabPFN 模型工件包含训练样本与标签上下文。
