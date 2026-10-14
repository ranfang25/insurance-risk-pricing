# 车险风险定价与客户分层项目（Windows 接手说明）

> 历史接手说明，仅供项目背景参考；当前运行方法与验证结果以根目录 README 为准。

## 1. 项目目标

复现简历中的个人项目：

> 基于可解释机器学习的车险风险定价与客户分层

使用 OpenML 上的公开法国机动车保险数据：

- `freMTPL2freq`：保单和出险频率数据，OpenML Data ID `41214`，约 67.8 万张保单。
- `freMTPL2sev`：赔款金额数据，OpenML Data ID `41215`。

最终需要产出一个可在 GitHub 展示、可在面试中演示、且简历描述与真实代码和结果一致的完整项目。

## 2. 当前状态（重要）

截至本 README 写入时，当前目录中只有空的 `data` 目录，没有可依赖的源代码、测试、虚拟环境或完整模型结果。之前的 Mac 运行在中途被停止，曾遇到以下问题：

- LightGBM 在 macOS 上需要 OpenMP 运行库。Windows 通常可直接使用 wheel，如出现 DLL 问题再安装 Microsoft Visual C++ Redistributable。
- OpenML 数据下载曾被中断，缓存可能不完整，Windows 上应重新下载。
- 之前环境曾出现 `ModuleNotFoundError: No module named 'narwhals._interchange'`，说明依赖版本不兼容。不要直接复用当时的松散依赖，应先固定经验证的版本。

因此，明天应当从“创建可复现的 Windows 项目”开始，不要假设已经完成端到端运行。

## 3. 明天要完成的工作

### A. 搭建可复现环境

1. 使用 Windows x64 版 Python 3.12。
2. 使用 `uv` 管理环境和锁定文件。
3. 固定并验证 `pandas`、`scikit-learn`、`narwhals`、`lightgbm`、`shap` 和 `streamlit` 的兼容版本，生成 `uv.lock`。
4. 创建 `pyproject.toml`、`.gitignore` 和标准项目目录。

建议结构：

```text
insurance-risk-pricing/
├── README.md
├── pyproject.toml
├── uv.lock
├── app.py
├── data/
│   ├── cache/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── tables/
├── src/insurance_pricing/
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
   - 平均赔付强度 `ClaimAmount / ClaimNb`
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
2. 设置清晰的示例定价规则，输出建议保费、风险加成和组合预期盈利；必须标明这是求职作品的模拟方案，不是真实保险报价。
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
4. 只有全量流程成功、结果经过人工审查后，才初始化 Git 并上传 GitHub。
5. 不上传虚拟环境、OpenML 原始缓存、过大的模型文件或包含本地绝对路径的文件。
6. GitHub 上传前再确认仓库名称以及公开/私有设置。

## 4. Windows PowerShell 建议操作顺序

在 Windows 上打开 PowerShell，进入项目目录。在代码和 `pyproject.toml` 创建完成后，依次执行：

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

# 启动展示仪表盘
uv run streamlit run app.py
```

如 OpenML 下载中断或缓存损坏，请先将对应缓存移至系统回收站，再重试 `quick` 模式。

## 5. 预期产出

完成后至少应有：

```text
models/
├── poisson_glm.joblib
├── gamma_glm.joblib
├── tweedie_glm.joblib
└── lightgbm_tweedie.txt

reports/
├── metrics.json
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
    └── policy_predictions_powerbi.csv
```

## 6. 完成标准

只有同时满足以下条件，才算项目完成：

- Windows 全新环境中 `uv sync --dev` 成功。
- 所有测试和静态检查通过。
- `quick` 和 `full` 模式均能从数据下载运行到报告输出。
- 全量模式使用约 67.8 万张保单，样本量由程序实际输出确认，不手工填写。
- 模型评估表不含无效值，图表可正常打开。
- Streamlit 能启动，Power BI 所需 CSV 字段完整。
- 所有简历数字和技术描述均可从实际运行结果中核对；若实际实现与简历文字不一致，应修改简历，不应伪造结果。
- GitHub README 包含项目背景、方法、实验结果、运行方法、限制和免责声明。

## 7. 给明天 Codex 的直接任务

可以直接把下面这句交给 Codex：

> 请按照本 README 从零完成车险风险定价项目。先在 Windows 上建立锁定依赖的 Python 3.12 环境，解决 `narwhals._interchange` 的版本兼容问题；然后实现数据处理、Poisson/Gamma/Tweedie GLM、LightGBM Tweedie、Gini/Lorenz/校准评估、SHAP、风险分层、PSI 和漂移/盈利模拟、Streamlit 仪表盘以及 Power BI CSV。先跑 quick，检查结果后再跑 full。未完成全量验证前不要上传 GitHub，也不要把未运行的结果写成已完成。
