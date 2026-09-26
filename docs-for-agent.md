# Featune 开发交接 / Developer handoff

Updated: 2026-09-23

先读 [设计与实施状态](IMPLEMENTATION_PLAN.md)、[贡献规范](CONTRIBUTING.md) 和[基准结论](benchmarks/README.md)。原需求中的设想已归入设计文档的已实现/待办清单，不再以聊天式需求作为当前实现事实。

Read the implementation status, contribution rules and benchmark conclusions before changing behavior. Proposed research goals are not shipped features.

- 默认固定 TabPFN v2，安装包含 Torch 与 Plotly；v3.5 必须用户显式选择并自行取得授权。
- 所有 sampler 使用封闭 FeatureSpec/IR；不执行生成代码，不访问 outer holdout 提议或选模。
- 算子、训练统计和预处理保持 fold-local；复用 compiler/evaluator 的共同路径。
- resume 严格校验实验指纹，不兼容 override 是归档重启。模型/缓存工件只从可信来源加载。
- 联合 HPO 是 trial 间的有限参数搜索；不把它描述成 epoch 中途改学习率、优化器或网络再续训。
- 类、方法和辅助函数使用英文 Google docstring，说明输入输出、副作用与异常；文件头使用 SPDX、职责说明、Created 日期。沿用现有命名和模块边界。
- 持续开发保留可运行测试与示例；付费 API 不进入默认测试。修改业务逻辑做相应验证，文档/清理改动优先检查链接、数据保留和发布文件范围。
- 原始实验 JSON/CSV、模型、缓存、下载数据和调试报告输出到 `runs/`；Git 仅留结论、Markdown 对比表、复现协议与必要图表。不要删除示例配置/示例数据。
- 当前证据是小样本多 seed 对比，未证实 LLM 稳定提升准确率或速度；可追溯不等于解释正确。
- 不引入单实现接口、重复工具类或“以后可能用”的抽象。新增能力按明确需求推进。

开发完成后更新设计中的状态和限制；不要只追加过时的测试数字而不说明当前验收范围。
