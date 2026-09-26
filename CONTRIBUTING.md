# Contributing / 贡献指南

Start with the [current design and implementation status](IMPLEMENTATION_PLAN.md) and [developer handoff](docs-for-agent.md). Update shipped behavior, validation scope and open work together when changing a contract; do not present a planned capability as implemented.

开发前先读[设计与实施状态](IMPLEMENTATION_PLAN.md)和[开发交接](docs-for-agent.md)。契约变更时同步更新已实现行为、验证范围和待办，不把规划能力写成当前事实。

Python source files begin with an English module description and creation date. Keep public concepts consistent: Schema → Proposal → FeatureCompiler → CVEvaluator → Trial → FeatureStudy. Follow the structure of the existing modules; avoid a second implementation of the same operation or evaluation path.

Python 文件使用英文模块说明与创建日期。公共概念保持 Schema → Proposal → FeatureCompiler → CVEvaluator → Trial → FeatureStudy 一致，不重复实现已有操作或评估路径。

Use the following header for every Python source file, including examples, benchmarks and tests. Preserve the original creation date; Git records authorship and subsequent changes. Keep the module summary and responsibility boundary specific to the file.

所有 Python 文件（包括示例、基准和测试）统一使用以下文件头。保留原始创建日期，作者与修改历史由 Git 管理；模块说明应明确职责和边界，不使用装饰性横幅。

```python
# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Summarize the module's responsibility in one sentence.

Describe its scope, important boundaries and side effects.

Created:
    YYYY-MM-DD
"""
```

Document every class, function and method, including private and nested helpers, with English PEP 257 docstrings in Google style. Start with a concise summary. Use `Args` for each argument except `self`/`cls`, `Returns` or `Yields` for outputs, `Raises` for relevant failure conditions, and `Notes` for mutation, persistence, caching, billing or other effects. Omit sections that do not apply; constructors describe initialized state instead of returning an object. Class `Attributes` describe meaningful configuration and runtime state. Specify accepted types, array shapes, units, defaults and edge cases where they affect the contract. Ruff enforces the Google docstring convention through the existing lint command.

所有类、函数和方法（包括私有及嵌套辅助函数）均采用英文 PEP 257 / Google 风格 docstring。说明输入、输出、重要异常和副作用；涉及数组时写明形状，涉及预算时写明单位，涉及可选参数时说明默认行为。类说明配置与运行时属性；不适用的章节省略，构造函数说明初始化状态。Ruff 在现有检查命令中校验该规范。

Inline comments explain why a non-obvious decision is necessary: leakage prevention, cache identity, state transitions, transaction boundaries or uncertain billing. Do not repeat the code mechanically. Source fingerprints include Python source text, so documentation-only changes can invalidate compatibility with older persisted studies; investigate the mismatch before explicitly opting into an incompatible restart.

行内注释解释关键决策的原因，例如防止泄漏、缓存身份、状态转换、事务边界和未知计费，不逐行复述代码。源码指纹包含 Python 源文件文本，因此仅修改文档也可能导致旧 study 的恢复兼容性检查失败；应先确认差异，再显式选择不兼容重启。

```bash
pip install -e '.[dev]'
ruff format featune tests examples benchmarks
ruff check featune tests examples benchmarks
pytest -q
python examples/quickstart.py
python -m build
```

New DSL operations need a closed signature, input/parameter checks, compiler implementation, a train/validation isolation check if stateful, and matching English/Chinese reference documentation. Do not execute model-generated code. Secrets and raw row values must never enter logs or committed artifacts.

新增 DSL 操作需要受控签名、类型/参数检查、编译器实现和双语文档；学习统计的操作需要验证训练/验证隔离。不执行 LLM 生成代码，不将密钥和原始数据行写入日志或提交文件。

Tests involving paid LLM calls are explicit examples/benchmarks, not part of CI. CI uses mocked HTTP. Benchmark changes must preserve an untouched outer test set, record failed trials and budgets, and disclose operator/model differences. Do not claim an improvement without published measurements.

Keep generated JSON/CSV, trial histories, databases, caches, models and HTML run reports locally under ignored `runs/`. Commit only reviewed Markdown conclusions/comparison tables, reproducibility notes and referenced figures. The benchmark runner still records raw evidence locally; publication summaries must retain failure counts and negative results, and explicitly disclose when raw records are no longer distributed. Maintained tests, example JSON configuration and example CSV inputs are source assets, not disposable test output.

原始实验工件放在 `runs/`；仓库仅保留 Markdown 结论/对比表、复现说明和必要图表。发布汇总必须保留失败数量和负向结论，并说明原始逐次记录是否仍可获取。测试源码、示例 JSON 配置及示例 CSV 输入不是测试垃圾，不应删除。

收费 LLM 调用只能显式执行，CI 使用模拟 HTTP。Benchmark 必须保留独立外层测试集、失败记录与预算信息，披露算子/模型差异；不发布未经测量的性能结论。

The benchmark extra pins scikit-learn below 1.6 for OpenFE 0.0.12's legacy mean_squared_error API. Use a separate virtual environment for that comparator; the core library is also tested on newer sklearn.

benchmark extra 为 OpenFE 0.0.12 固定兼容的 sklearn 范围，建议使用独立虚拟环境；核心库另行验证新版本 sklearn。

The ordinary test suite mocks the default model factory when executing README examples. Run `FEATUNE_TEST_TABPFN=1 OMP_NUM_THREADS=2 pytest tests/test_tabpfn.py -q` for actual pinned-checkpoint classification/regression, cached resume and offline fresh-process export checks. Default integration tests use public pinned v2 weights and run in CI. Set `FEATUNE_TEST_TABPFN_VERSION=v3.5` to test the optional preset instead; this requires separate license acceptance and model access. Interactive terminals open the official login flow and continue downloading after the callback. For headless CI, set `TABPFN_TOKEN`, `TABPFN_NO_BROWSER=1` and `HF_TOKEN` if needed. Optional 3.5 CI runs only via workflow_dispatch with `tabpfn_integration=true`, using the protected `tabpfn-integration` environment. Never expose credentials to untrusted pull-request code or publish gated checkpoint caches. A skipped job does not establish real-weight acceptance.
