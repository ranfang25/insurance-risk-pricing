"""暴露期加权的模型评估与校准。"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def normalized_gini(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    """按预测单位暴露期成本排序的暴露期加权标准化 Gini。"""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if len(actual) != len(predicted) or len(actual) != len(weight):
        raise ValueError("长度不一致")
    if np.any(weight <= 0) or np.any(actual < 0):
        raise ValueError("赔款及暴露期需为非负值，暴露期须大于零")
    total_loss = np.sum(actual * weight)
    if total_loss <= 0:
        return float("nan")

    def area(score: np.ndarray) -> float:
        order = np.argsort(score, kind="stable")
        x = np.r_[0, np.cumsum(weight[order]) / weight.sum()]
        y = np.r_[0, np.cumsum((actual * weight)[order]) / total_loss]
        return float(np.trapezoid(y, x))

    perfect = area(actual)
    denominator = 0.5 - perfect
    return (0.5 - area(predicted)) / denominator if denominator > 0 else float("nan")


def lorenz_points(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> pd.DataFrame:
    order = np.argsort(predicted, kind="stable")
    exposure = np.asarray(weight)[order]
    loss = np.asarray(actual)[order] * exposure
    return pd.DataFrame(
        {
            "exposure_share": np.r_[0, np.cumsum(exposure) / exposure.sum()],
            "loss_share": np.r_[0, np.cumsum(loss) / loss.sum()],
        }
    )


def score_models(frame: pd.DataFrame, columns: list[str], power: float = 1.5) -> pd.DataFrame:
    rows = []
    for column in columns:
        prediction = frame[column].to_numpy()
        rows.append(
            {
                "model": column,
                "mean_tweedie_deviance": mean_tweedie_deviance(
                    frame["PurePremium"], prediction, sample_weight=frame["Exposure"], power=power
                ),
                "normalized_gini": normalized_gini(
                    frame["PurePremium"], prediction, frame["Exposure"]
                ),
                "observed_loss": frame["ClaimAmount"].sum(),
                "predicted_loss": float(np.sum(prediction * frame["Exposure"])),
            }
        )
    return pd.DataFrame(rows)


def assign_tiers(reference: pd.Series, values: pd.Series, n: int = 5) -> pd.Series:
    """用训练集分位点定义稳定风险层级。"""
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, n + 1)[1:-1]))
    return pd.Series(np.searchsorted(cuts, values, side="right") + 1, index=values.index)


def calibration(frame: pd.DataFrame, score: str, n: int = 10) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["decile"] = pd.qcut(ranked[score].rank(method="first"), n, labels=False) + 1
    table = ranked.groupby("decile", as_index=False).agg(
        policies=("IDpol", "size"),
        exposure=("Exposure", "sum"),
        actual_claims=("ClaimAmount", "sum"),
        predicted_cost=("predicted_cost", "sum"),
    )
    table["actual_pure_premium"] = table["actual_claims"] / table["exposure"]
    table["predicted_pure_premium"] = table["predicted_cost"] / table["exposure"]
    return table
