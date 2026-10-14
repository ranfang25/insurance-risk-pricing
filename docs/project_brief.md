# 车险风险定价与客户分层：项目设计与验证说明

本文件记录项目的建模目标、实现范围和验证要点。运行命令与详细结果以根目录 [README](../README.md) 为准。

## 1. 项目目标

建立一个基于可解释机器学习的车险风险定价与客户分层流程。

使用 OpenML 上的公开法国机动车保险数据：

- `freMTPL2freq`：保单和出险频率数据，OpenML Data ID `41214`，约 67.8 万张保单。
- `freMTPL2sev`：赔款金额数据，OpenML Data ID `41215`。

项目从公开数据获取和清洗开始，完成建模、评估、风险分层、模拟定价与监控，并提供可复现的代码和结果。

## 2. 实现状态

项目已在 Windows、Python 3.12 和 `uv.lock` 锁定的环境中完成 `quick` 与 `full` 两种运行。全量流程使用 678,013 张有效保单；测试、静态检查和格式检查均已通过。模型结果、图表和汇总表分别保存在 `output/quick/` 与 `output/full/`。

## 3. 设计与实现

### A. 搭建可复现环境

1. 使用 Windows x64 版 Python 3.12。
2. 使用 `uv` 管理环境和锁定文件。
3. 固定并验证 `pandas`、`scikit-learn`、`narwhals`、`lightgbm`、`shap` 和 `streamlit` 的兼容版本，生成 `uv.lock`。
4. 创建 `pyproject.toml`、`.gitignore` 和标准项目目录。

当前结构：

```text
insurance_test/
├── README.md
├── pyproject.toml
├── uv.lock
├── app.py
├── input/
│   ├── cache/
│   └── processed/
├── output/
│   ├── quick/
│   └── full/
├── code/insurance_pricing/
│   ├── data.py
│   ├── features.py
│   ├── metrics.py
│   ├── monitoring.py
│   └── pipeline.py
└── tests/
```

### B. 数据处理与指标构建

1. 通过 OpenML ID 下载频率数据和赔款数据，并保存本地缓存。
2. 按 `IDpol` 关联保单和赔款，同一保单的多笔赔款先汇总。
3. 完成类型处理、缺失检查、异常值处理和暴露期校正。
4. 构建以下核心指标：
   - 出险频率 `ClaimNb / Exposure`
   - 有赔款保单的平均赔付强度 `ClaimAmount / ClaimRecords`
   - 纯风险保费 `ClaimAmount / Exposure`
   - 赔付率代理指标和保单风险分层

### C. 建模

1. 按照可复现的随机种子划分训练集和测试集。
2. 建立频率模型：Poisson GLM，使用 `Exposure` 作为权重或 offset。
3. 建立赔款强度模型：Gamma GLM，只在有赔款的样本上训练。
4. 建立 Tweedie GLM，直接预测单位暴露期的纯风险保费。
5. 建立 LightGBM Tweedie 非线性模型。
6. 所有模型需保存预测结果、训练参数和随机种子。

### D. 模型评估与可解释性

至少输出：

- Mean Tweedie Deviance
- Normalized Gini 系数
- Lorenz 曲线
- 风险十分位的实际值与预测值校准表/校准图
- LightGBM 的 SHAP 全局特征重要性
- 驾驶员年龄、车辆特征、地区密度和历史风险等级等特征对预期赔付成本的影响分析

### E. 客户分层、定价和监控

1. 按预测纯风险保费将客户划分为 5 个风险层。
2. 设置明确的模拟定价规则，输出建议保费、风险加成和组合预期盈利；这些指标基于公开数据与假设，不是实际保险报价。
3. 建立 PSI 监控，覆盖关键特征和预测风险分布。
4. 模拟客户结构变化，输出分层实际赔付率、定价偏差和组合盈利影响。

### F. 展示层

1. 用 Streamlit 完成可本地演示的交互式仪表盘。
2. 另外输出适合 Power BI Desktop 直接导入的 CSV：
   - `policy_predictions_powerbi.csv`
   - `model_metrics.csv`
   - `risk_calibration.csv`
   - `risk_tier_summary.csv`
   - `monitoring_summary.csv`
   - `shap_importance.csv`
3. Power BI 建议页面：组合概览、风险分层、模型表现、SHAP 解释、漂移监控。

### G. 测试、质量和 GitHub

1. 为 Gini、PSI、分层和数据关联编写单元测试。
2. 运行代码格式和静态检查。
3. 先运行 `quick` 模式，再运行全量数据。
4. 全量流程运行成功且结果经过审查后再提交 GitHub。
5. 不上传虚拟环境、OpenML 原始缓存、过大的模型文件或包含本地绝对路径的文件。
6. 公开仓库仅包含可复现代码、文档和适合公开的汇总结果。

## 4. Windows PowerShell 运行顺序

在 Windows 上打开 PowerShell，进入项目目录后依次执行：

```powershell
# 安装 uv（如尚未安装）
winget install --id=astral-sh.uv -e

# 检查 Python 和 uv
py -3.12 --version
uv --version

# 创建环境并安装锁定依赖
uv sync --dev

# 运行测试和静态检查
uv run pytest -q
uv run ruff check .

# 先用小样本检查端到端流程
uv run python -m insurance_pricing.pipeline --mode quick

# quick 成功后运行全量数据
uv run python -m insurance_pricing.pipeline --mode full

# 启动交互式分析应用
uv run streamlit run app.py
```

如 OpenML 下载中断或缓存损坏，请先将对应缓存移至系统回收站，再重试 `quick` 模式。

## 5. 输出

两种运行模式分别生成：

```text
output/{quick,full}/
├── metrics.json
├── model_params.json
├── models/
├── figures/
│   ├── lorenz_curve.png
│   ├── calibration.png
│   ├── shap_summary.png
│   └── monitoring.png
└── tables/
    ├── model_metrics.csv
    ├── risk_calibration.csv
    ├── risk_tier_summary.csv
    ├── monitoring_summary.csv
    ├── shap_importance.csv
    ├── feature_effects.csv
    ├── scenario_summary.csv
    └── policy_predictions_powerbi.csv
```

## 6. 验证记录与边界

- 本机 `uv sync --dev`、4 项单元测试、Ruff 静态检查与格式检查通过。
- `quick` 与 `full` 模式均完成，图表和汇总表已生成；全量样本量与模型指标见根目录 README 和 `output/full/metrics.json`。
- Streamlit 五个页面执行时无异常。保单级 CSV、模型文件和 OpenML 原始缓存因体积较大，不纳入 Git。
- 公开数据不含真实保费、费用或业务承保决策；利润和赔付率仅为基于假设保费的代理指标。随机划分和模拟漂移也不代表真实业务上线后的表现。
