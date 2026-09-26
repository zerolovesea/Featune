# 实验环境 / Environment

执行日期：2026-09-22–23；以下版本为执行时记录，不代表自动跟随最新版本。

Python 3.11.14 · Darwin 27.0.0 · arm64 · Apple M4 Pro · 24 GiB RAM

CPU 顺序运行，OMP_NUM_THREADS=2；权重预热后计时，HF_HUB_OFFLINE=1，TABPFN_NO_BROWSER=1，TABPFN_EXCLUDE_DEVICES=cuda,mps。

LLM 使用已配置的 Anthropic-compatible 服务，模型标识 claude-opus-4-8。密钥、服务私有地址和本机绝对路径不在文档中保存。

## 已安装版本 / Installed versions

```text
Jinja2==3.1.6
MarkupSafe==3.0.3
PyYAML==6.0.3
annotated-types==0.8.0
anyio==4.15.1
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
cloudpickle==3.1.2
contourpy==1.3.3
cycler==0.12.1
einops==0.8.2
et_xmlfile==2.0.0
featune==1.0.0
filelock==4.0.1
fonttools==4.65.0
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.32.0
idna==3.20
joblib==1.6.0
kiwisolver==1.5.1
lightgbm==4.7.0
matplotlib==3.11.2
mlx==0.32.2
mlx-metal==0.32.2
mpmath==1.3.0
narwhals==2.26.0
networkx==3.6.1
numpy==1.26.4
openfe==0.0.12
openpyxl==3.1.5
packaging==26.3
pandas==2.2.3
pillow==12.3.0
plotly==7.1.0
pyarrow==25.0.1
pydantic==2.13.5
pydantic-settings==2.15.0
pydantic_core==2.46.5
pydot==4.0.1
pyparsing==3.3.3
python-dateutil==2.9.0.post0
python-dotenv==1.2.3
pytz==2026.3.post1
requests==2.34.2
safetensors==0.8.0
scikit-learn==1.5.2
scipy==1.17.1
setuptools==84.0.0
six==1.17.0
skrub==0.10.1
sympy==1.14.0
tabpfn==9.0.0
threadpoolctl==3.7.0
torch==2.14.0
tqdm==4.70.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.4
urllib3==2.8.0
```

实验配置见 [PROTOCOL.md](PROTOCOL.md)；权重 revision 和摘要见 [模型适配器](../../../featune/tabpfn.py)。重新执行会生成新的逐次记录，LLM 输出不保证跨时间完全一致。
