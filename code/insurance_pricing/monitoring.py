"""特征漂移与模拟组合变化。"""

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """使用参考分布分箱计算 Population Stability Index。"""
    reference = np.asarray(reference)
    current = np.asarray(current)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI 输入不能为空")
    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.quantile(reference.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0 if np.all(current == reference[0]) else float("inf")
        ref_bins = np.digitize(reference.astype(float), edges[1:-1])
        cur_bins = np.digitize(current.astype(float), edges[1:-1])
        labels = np.arange(len(edges) - 1)
    else:
        ref_bins = reference.astype(str)
        cur_bins = current.astype(str)
        labels = np.union1d(ref_bins, cur_bins)
    p = np.array([(ref_bins == label).mean() for label in labels])
    q = np.array([(cur_bins == label).mean() for label in labels])
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def monitoring_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    columns = ["DrivAge", "VehAge", "BonusMalus", "Density", "predicted_pure_premium"]
    return pd.DataFrame(
        [{"feature": col, "psi": psi(reference[col], current[col])} for col in columns]
    )


def price_and_segment(
    frame: pd.DataFrame,
    reference_score: pd.Series,
    expense_ratio: float = 0.20,
    risk_loading: float = 0.30,
) -> pd.DataFrame:
    from insurance_pricing.metrics import assign_tiers

    out = frame.copy()
    out["risk_tier"] = assign_tiers(reference_score, out["predicted_pure_premium"])
    out["suggested_annual_premium"] = out["predicted_pure_premium"] * (1 + risk_loading)
    out["written_premium"] = out["suggested_annual_premium"] * out["Exposure"]
    out["predicted_cost"] = out["predicted_pure_premium"] * out["Exposure"]
    out["expected_profit"] = (1 - expense_ratio) * out["written_premium"] - out["predicted_cost"]
    out["realized_profit_proxy"] = (1 - expense_ratio) * out["written_premium"] - out["ClaimAmount"]
    return out


def tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.groupby("risk_tier", as_index=False).agg(
        policies=("IDpol", "size"),
        exposure=("Exposure", "sum"),
        observed_loss=("ClaimAmount", "sum"),
        predicted_cost=("predicted_cost", "sum"),
        written_premium=("written_premium", "sum"),
        expected_profit=("expected_profit", "sum"),
        realized_profit_proxy=("realized_profit_proxy", "sum"),
    )
    table["observed_loss_ratio"] = table["observed_loss"] / table["written_premium"]
    table["pricing_bias"] = table["observed_loss"] / table["predicted_cost"] - 1
    return table


def shift_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """按风险层级改变组合权重，展示结构变化的财务影响。"""
    high = frame["risk_tier"] >= 4
    multiplier = np.where(high, 1.5, 0.75)
    rows = []
    for name, weights in [("baseline", np.ones(len(frame))), ("high_risk_shift", multiplier)]:
        premium = np.sum(weights * frame["written_premium"])
        actual = np.sum(weights * frame["ClaimAmount"])
        predicted = np.sum(weights * frame["predicted_cost"])
        rows.append(
            {
                "scenario": name,
                "weighted_policies": weights.sum(),
                "observed_loss_ratio": actual / premium,
                "pricing_bias": actual / predicted - 1,
                "expected_profit": 0.8 * premium - predicted,
                "realized_profit_proxy": 0.8 * premium - actual,
            }
        )
    return pd.DataFrame(rows)
