# Featune 对比实验设计与状态 / Comparison design and status

Updated: 2026-09-23

## 目的与边界 / Questions

分别检验特征搜索的外层指标改善、字段语义的额外价值、搜索的资源成本和解释可核查性。不能把模型替换收益归于框架，不能以“挑出能赢的数据集”为评价原则。新实验先固定任务、seed、预算和指标，再运行全部方法。

Separate feature-search gains, semantic effects, model replacement and resource cost. Preserve neutral/negative outcomes in aggregate conclusions.

## 已完成 / Completed

| 实验 | 配置与规模 | 证据等级 | 保留位置 |
|---|---|---|---|
| 固定 v2 | 六数据集、五 seed、最多 1,200 行、240 runs | intermediate | [报告](results/tabpfn-v2/report.md)、[汇总表](results/tabpfn-v2/comparisons.md) |
| 传统模型公开对照 | 全量 Adult/California，三 seed，48 runs | intermediate；不同于 v2 子样本 | [历史汇总](results/HISTORICAL.md) |
| Adult 语义/匿名/错配 | 线性模型，两 seed，10 runs | intermediate；非稳定语义优势证明 | 历史汇总 |
| CoverType 受限 OpenFE | 全量数据，线性模型，单 seed | 单次对照 | 历史汇总 |
| 大字段上下文 | 200/500/1000 字段、三 seed、五策略，共 45 组离线运行 | smoke；未验证真实 LLM 收益 | 历史汇总 |
| 自主搜索与语义检查 | Wine/Digits 新外层划分；三项真实 LLM 元数据审核；200/500/1000 字段真实 LLM 单 seed 消融 | 初步；结果含持平和负向运行 | [初步结果](results/autonomous-20260923.md) |

旧 Retail 全量目标截尾结果撤回；新 v2 采用原始 Quantity 的五 seed 子样本对照已经完成，不能继续写作待运行。OpenFE 的混合类别 Feather 问题已通过比较器内字符串化处理，旧 sklearn API 通过隔离的 sklearn 1.5.2 环境兼容；这些不再是本轮未完成的阻塞项。

本轮 `diabetes` 是 sklearn 的 442 行回归任务，不是 OpenML/Pima 的二分类 diabetes，不可混用数据集名和结果。

## 当前协议 / Executed protocol

完整预声明参数见 [PROTOCOL.md](results/tabpfn-v2/PROTOCOL.md)，执行环境见 [ENVIRONMENT.md](results/tabpfn-v2/ENVIRONMENT.md)。同组共享数据子集和外层 split，25% holdout、3-fold CV；提议、选模和停止只看外层训练数据。冻结后主指标/次指标及解释可访问 holdout，但不回流搜索。

方法为原始 v2、Random、Evolution、语义 LLM、匿名 LLM、受限 OpenFE；原始 Linear/HistGradientBoosting 单独衡量模型选择。Featune 四次提议、每次最多一列，OpenFE 32 个逐行候选、最多十列；内部拟合预算不等价。失败和预算阻止不通过补跑消除。

主指标：二分类 AUC、多分类 weighted OVR AUC、回归 RMSE。保留均值/标准差、配对改善与描述性 95% bootstrap 区间、胜/平次数、成本、解释耗时、提议状态。五 seed 的区间不是多重比较校正后的普遍优越性证据。

## 当前判断 / Conclusions

语义 LLM 六个任务的增益区间均包含零；语义/匿名差值区间也均包含零。Adult Random 五 seed 全部改善，是一个局部正向结果。语义 LLM 搜索及主评分约为原始 v2 基线的 3.44–17.70 倍。表达式和来源可核查，但部分自然语言理由不准确。不能预先承诺精度、速度或语义收益。

No full benchmark evidence of broad superiority has been established. The measured benefit is constrained, traceable search with task-dependent score changes.

## 尚未执行 / Open experiments

1. 在更多任务和 seed 上做严格等实际拟合次数、等派生列数量的对照；新 Wine/Digits 已报告共同拟合上限和预算曲线，实际拟合次数尚不相同。
2. 将已完成的真实 LLM 单 seed wide-context 消融扩展到多 seed 和真实宽表；记录实际 token、检索时间、提议有效率和外层增益。
3. 独立语义分类任务，例如 German Credit/Contraceptive Method Choice；当前未执行，不引用其假想成绩。CAAFE 也未纳入实测矩阵。
4. TabPFN v3.5 完整真实权重集成与性能验收；需要用户显式选择和许可，不混用 v2 证据。

## 结果保留 / Retention

执行时保留本地全部记录，以便失败诊断和配对校验；发布到仓库时只保留 Markdown 结论、汇总对比表、协议/环境及必要图表。原始 JSON/CSV 已按项目要求清理，不再宣称仓库持有完整逐次审计记录。重跑输出到 `runs/`，不要覆盖已发布结论后挑选有利结果。
