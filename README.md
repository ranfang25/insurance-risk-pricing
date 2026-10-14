# 车险风险定价与客户分层

本项目基于 OpenML 法国机动车保险公开数据，完整实现从保单与赔款关联、频率和赔款强度建模，到纯风险保费预测、风险分层、模拟定价、漂移监控及交互式分析的可复现流程。项目设计与验证说明见 [docs/project_brief.md](docs/project_brief.md)。

## 数据和方法

- 数据：OpenML [`freMTPL2freq`](https://www.openml.org/d/41214)（ID 41214）与 [`freMTPL2sev`](https://www.openml.org/d/41215)（ID 41215）。多笔赔款先按 `IDpol` 汇总，再与保单做一对一关联。
- 目标：出险频率为 `ClaimNb / Exposure`；有赔款保单的平均赔付强度为 `ClaimAmount / ClaimRecords`；纯风险保费为 `ClaimAmount / Exposure`。暴露期截到一年，缺失和明显无效的保单字段剔除；实际赔款金额不截断，以保留尾部风险。频率表的出险笔数多于赔款表中可关联的赔款记录；双阶段预测乘以训练集的可关联赔款记录比例，比例及两类计数均写入 `metrics.json`。
- 随机种子固定为 42，保单按 80%/20% 划分训练和测试。频率用 Poisson GLM，强度用 Gamma GLM，两者乘积形成双阶段估计；另用 Tweedie GLM 和 LightGBM Tweedie 直接预测纯风险保费。模型使用暴露期或赔款记录数作为训练权重。LightGBM 预测另乘以训练集总赔款与总预测成本之比，作为整体水平校准；系数写入 `metrics.json`，测试集不参与校准。
- 评估：测试集的暴露期加权 Mean Tweedie Deviance、Normalized Gini、Lorenz 曲线和风险十分位校准。SHAP 解释 LightGBM 对数成本尺度的模型输出。
- 定价：按训练集 LightGBM 预测分位点划分五层，模拟年度建议保费为预测纯风险保费的 1.3 倍；费用率假设为 20%。组合结构变化模拟增加高风险客户权重，利润结果仅用于情景比较。

## 运行

需要 Windows x64、Python 3.12、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和网络连接。项目根目录执行：

```powershell
uv sync --dev
uv run python -m pytest -q
uv run python -m ruff check .
uv run python -m ruff format --check app.py code tests
uv run python -m insurance_pricing.pipeline --mode quick
uv run python -m insurance_pricing.pipeline --mode full
uv run streamlit run app.py
```

首次运行会按 OpenML ID 下载数据，原始压缩文件保存在 `input/cache/`，解析后数据保存在 `input/processed/`。`quick` 模式从完整数据随机抽取最多 30,000 张保单，用于检查流程；`full` 模式使用全部有效保单。输出分别保存在 `output/quick/` 和 `output/full/`，避免覆盖。Power BI Desktop 可直接导入 `output/full/tables/` 中的 CSV。

## 项目目录

```text
code/insurance_pricing/  数据、特征、指标、监控、建模代码
input/cache/             OpenML 原始及本地缓存，不进入 Git
input/processed/         解析后的本地数据，不进入 Git
output/quick/            快速检查结果
output/full/             全量结果
tests/                   核心计算和关联测试
docs/project_brief.md    项目设计与验证说明
app.py                   Streamlit 交互式分析应用
```

每次运行都会输出模型、`metrics.json`（包含随机种子、校准系数与风险层阈值）、`model_params.json`、图表和以下 CSV：`model_metrics.csv`、`risk_calibration.csv`、`risk_tier_summary.csv`、`monitoring_summary.csv`、`shap_importance.csv`、`feature_effects.csv`、`scenario_summary.csv`、`policy_predictions_powerbi.csv`。大体积原始缓存、模型和保单级导出默认不纳入 Git。

Power BI 建议做五页：组合概览、风险分层、模型表现、SHAP 解释、漂移监控；其中 `policy_predictions_powerbi.csv` 用于保单级筛选，其余汇总表直接用于图表。

## 实验结果

本机已在 Python 3.12.14 与 `uv.lock` 锁定环境中运行 `quick` 和 `full`；全量数据为 678,013 张保单，按固定随机种子划分后，测试集为 135,603 张保单。4 项单元测试、Ruff 静态检查与格式检查均通过，Streamlit 五个页面执行时无异常。以下为全量测试集实际结果。

| 模型 | Mean Tweedie Deviance ↓ | Normalized Gini ↑ | 预测赔款 | 观察赔款 |
| --- | ---: | ---: | ---: | ---: |
| Poisson × Gamma | 85.163 | 0.174 | 11,116,238 | 11,868,104 |
| Tweedie GLM | 84.312 | 0.263 | 11,900,290 | 11,868,104 |
| LightGBM Tweedie（训练集校准后） | 82.125 | 0.328 | 11,921,397 | 11,868,104 |

LightGBM 在本次随机测试划分上的偏差指标和排序指标均优于两个 GLM 基线；十分位校准图显示中间分位仍存在局部偏差，不能只凭总体赔款接近就认为各客群已准确校准。SHAP 全局重要性中，`BonusMalus`、驾驶员年龄、车龄、地区密度排名靠前；分箱影响见 `feature_effects.csv`，只反映模型关联，不应解释为因果关系。文中的样本量和指标均可在 `output/full/` 中核对。

## 限制

公开数据不包含真实保费、费用及业务承保决策，因此赔付率和利润仅为基于假设保费的代理指标。随机保单划分不能代表跨时期外推；情景漂移是人为重加权，不是实际发生的市场变化。SHAP 展示模型关联，不代表因果效应。
