"""本地演示仪表盘。"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="车险风险定价与客户分层", layout="wide")
st.title("车险风险定价与客户分层")
st.caption("求职作品中的模拟定价方案，并非真实保险报价。")
mode = st.sidebar.selectbox("运行结果", ["full", "quick"])
folder = ROOT / "output" / mode
if not (folder / "metrics.json").exists():
    st.info(f"请先执行 uv run python -m insurance_pricing.pipeline --mode {mode}")
    st.stop()

tables = folder / "tables"
pred = pd.read_csv(tables / "policy_predictions_powerbi.csv")
metrics = pd.read_csv(tables / "model_metrics.csv")
tiers = pd.read_csv(tables / "risk_tier_summary.csv")
monitor = pd.read_csv(tables / "monitoring_summary.csv")
shap = pd.read_csv(tables / "shap_importance.csv")
effects = pd.read_csv(tables / "feature_effects.csv")
scenario = pd.read_csv(tables / "scenario_summary.csv")

first, second, third = st.columns(3)
first.metric("测试集保单数", f"{len(pred):,}")
second.metric("观察赔款", f"{pred['ClaimAmount'].sum():,.0f}")
third.metric("模拟预期盈利", f"{pred['expected_profit'].sum():,.0f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["组合概览", "风险分层", "模型表现", "漂移监控", "保单探索"])
with tab1:
    st.dataframe(scenario, hide_index=True, width="stretch")
    st.caption("结构变化通过增加高风险客户权重模拟；观察利润是测试集代理值。")
with tab2:
    st.dataframe(tiers, hide_index=True, width="stretch")
    st.bar_chart(tiers.set_index("risk_tier")[["observed_loss", "predicted_cost"]])
with tab3:
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.image(str(folder / "figures" / "lorenz_curve.png"))
    st.image(str(folder / "figures" / "calibration.png"))
    st.subheader("SHAP 全局重要性")
    st.dataframe(shap.head(20), hide_index=True, width="stretch")
    st.subheader("关键特征影响")
    chosen = st.selectbox("选择特征", effects["feature"].unique())
    st.dataframe(effects[effects["feature"] == chosen], hide_index=True, width="stretch")
    st.caption("SHAP 值位于模型的对数成本尺度，仅说明关联及模型贡献。")
with tab4:
    st.dataframe(monitor, hide_index=True, width="stretch")
    st.image(str(folder / "figures" / "monitoring.png"))
with tab5:
    selected_tiers = st.multiselect(
        "风险层", sorted(pred["risk_tier"].unique()), default=[1, 2, 3, 4, 5]
    )
    selected_regions = st.multiselect("地区", sorted(pred["Region"].unique()))
    filtered = pred[pred["risk_tier"].isin(selected_tiers)]
    if selected_regions:
        filtered = filtered[filtered["Region"].isin(selected_regions)]
    st.caption(f"当前筛选：{len(filtered):,} 张保单；展示前 1,000 行。")
    st.dataframe(
        filtered[
            [
                "IDpol",
                "Region",
                "risk_tier",
                "Exposure",
                "predicted_pure_premium",
                "suggested_annual_premium",
                "ClaimAmount",
            ]
        ].head(1000),
        hide_index=True,
        width="stretch",
    )
