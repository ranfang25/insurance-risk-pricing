"""从 OpenML 原始数据到模型、报表和图表的完整流程。"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split

from insurance_pricing.data import FEATURES, load_policies
from insurance_pricing.features import clean_predictions, make_encoder
from insurance_pricing.metrics import calibration, lorenz_points, score_models
from insurance_pricing.monitoring import (
    monitoring_table,
    price_and_segment,
    shift_summary,
    tier_summary,
)

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
SEED = 42
POWER = 1.5


def _save_plot(path: Path, x, ys: list[tuple], xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label in ys:
        ax.plot(x, values, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(mode: str) -> dict:
    input_dir = ROOT / "input" / "cache"
    result_dir = ROOT / "output" / mode
    model_dir = result_dir / "models"
    fig_dir = result_dir / "figures"
    table_dir = result_dir / "tables"
    for directory in [model_dir, fig_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    policies = load_policies(input_dir)
    full_rows = len(policies)
    data_quality = policies.attrs["quality"]
    if mode == "quick":
        policies = policies.sample(n=min(30000, full_rows), random_state=SEED).copy()
    train, test = train_test_split(policies, test_size=0.2, random_state=SEED)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    encoder = make_encoder()
    x_train = encoder.fit_transform(train[FEATURES])
    x_test = encoder.transform(test[FEATURES])
    joblib.dump(encoder, model_dir / "feature_encoder.joblib")

    poisson = PoissonRegressor(alpha=0.1, max_iter=150)
    poisson.fit(x_train, train["Frequency"], sample_weight=train["Exposure"])
    positive = (train["ClaimRecords"] > 0) & (train["Severity"] > 0)
    gamma = GammaRegressor(alpha=0.1, max_iter=150)
    gamma.fit(
        x_train[positive.to_numpy()],
        train.loc[positive, "Severity"],
        sample_weight=train.loc[positive, "ClaimRecords"],
    )
    matched_claim_ratio = train["ClaimRecords"].sum() / train["ClaimNb"].sum()
    tweedie = TweedieRegressor(power=POWER, alpha=0.1, max_iter=150)
    tweedie.fit(x_train, train["PurePremium"], sample_weight=train["Exposure"])
    for name, model in [("poisson_glm", poisson), ("gamma_glm", gamma), ("tweedie_glm", tweedie)]:
        joblib.dump(model, model_dir / f"{name}.joblib")

    lightgbm = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=POWER,
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=80,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )
    lightgbm.fit(x_train, train["PurePremium"], sample_weight=train["Exposure"])
    lightgbm.booster_.save_model(str(model_dir / "lightgbm_tweedie.txt"))

    raw_train_score = clean_predictions(lightgbm.predict(x_train))
    calibration_factor = train["ClaimAmount"].sum() / np.sum(raw_train_score * train["Exposure"])
    train_score = raw_train_score * calibration_factor
    scored = test.copy()
    scored["predicted_frequency"] = clean_predictions(poisson.predict(x_test))
    scored["predicted_severity"] = clean_predictions(gamma.predict(x_test))
    scored["poisson_gamma"] = clean_predictions(
        scored["predicted_frequency"] * scored["predicted_severity"] * matched_claim_ratio
    )
    scored["tweedie_glm"] = clean_predictions(tweedie.predict(x_test))
    scored["lightgbm_tweedie"] = clean_predictions(lightgbm.predict(x_test) * calibration_factor)
    scored["predicted_pure_premium"] = scored["lightgbm_tweedie"]
    scored = price_and_segment(scored, pd.Series(train_score))
    model_columns = ["poisson_gamma", "tweedie_glm", "lightgbm_tweedie"]
    metrics = score_models(scored, model_columns, POWER)
    calibration_table = calibration(scored, "predicted_pure_premium")
    tiers = tier_summary(scored)
    shifted = shift_summary(scored)
    reference = train[["DrivAge", "VehAge", "BonusMalus", "Density"]].copy()
    reference["predicted_pure_premium"] = train_score
    monitor = monitoring_table(reference, scored)

    # 将高风险样本加权复制，构造模拟的组合结构漂移。
    high = scored[scored["risk_tier"] >= 4].sample(frac=0.5, random_state=SEED)
    shifted_monitor = monitoring_table(reference, pd.concat([scored, high], ignore_index=True))
    shifted_monitor["scenario"] = "high_risk_shift"
    monitor["scenario"] = "baseline"
    monitor = pd.concat([monitor, shifted_monitor], ignore_index=True)

    names = encoder.get_feature_names_out()
    sample_count = min(1000, len(test))
    sample_indices = np.random.default_rng(SEED).choice(len(test), sample_count, replace=False)
    shap_values = shap.TreeExplainer(lightgbm).shap_values(x_test[sample_indices])
    if hasattr(shap_values, "toarray"):
        shap_values = shap_values.toarray()
    shap_values = np.asarray(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": names, "mean_abs_shap_log_cost": mean_abs})
    importance = importance.sort_values("mean_abs_shap_log_cost", ascending=False)
    sample = test.iloc[sample_indices].reset_index(drop=True)
    effects = []
    for feature in ["DrivAge", "VehAge", "VehPower", "Density", "BonusMalus"]:
        column_index = list(names).index(f"numeric__{feature}")
        grouped = pd.DataFrame(
            {
                "value": sample[feature],
                "shap": shap_values[:, column_index],
                "predicted_cost": scored.iloc[sample_indices]["predicted_pure_premium"].to_numpy(),
            }
        )
        grouped["bin"] = pd.qcut(grouped["value"].rank(method="first"), 5, labels=False) + 1
        part = grouped.groupby("bin", as_index=False).agg(
            policies=("value", "size"),
            mean_value=("value", "mean"),
            mean_shap_log_cost=("shap", "mean"),
            mean_predicted_cost=("predicted_cost", "mean"),
        )
        part.insert(0, "feature", feature)
        effects.append(part)
    feature_effects = pd.concat(effects, ignore_index=True)

    table_map = {
        "model_metrics": metrics,
        "risk_calibration": calibration_table,
        "risk_tier_summary": tiers,
        "monitoring_summary": monitor,
        "shap_importance": importance,
        "feature_effects": feature_effects,
        "scenario_summary": shifted,
    }
    for name, table in table_map.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8")
    export_columns = [
        "IDpol",
        "Exposure",
        "ClaimNb",
        "ClaimRecords",
        "ClaimAmount",
        "Frequency",
        "Severity",
        "PurePremium",
        *FEATURES,
        "predicted_frequency",
        "predicted_severity",
        *model_columns,
        "predicted_pure_premium",
        "risk_tier",
        "suggested_annual_premium",
        "written_premium",
        "predicted_cost",
        "expected_profit",
        "realized_profit_proxy",
    ]
    scored[export_columns].to_csv(
        table_dir / "policy_predictions_powerbi.csv", index=False, encoding="utf-8"
    )

    lorenz = lorenz_points(
        scored["PurePremium"], scored["predicted_pure_premium"], scored["Exposure"]
    )
    _save_plot(
        fig_dir / "lorenz_curve.png",
        lorenz["exposure_share"],
        [(lorenz["loss_share"], "Model"), (lorenz["exposure_share"], "Equality")],
        "Cumulative exposure",
        "Cumulative claim cost",
    )
    _save_plot(
        fig_dir / "calibration.png",
        calibration_table["decile"],
        [
            (calibration_table["actual_pure_premium"], "Observed"),
            (calibration_table["predicted_pure_premium"], "Predicted"),
        ],
        "Predicted risk decile",
        "Pure premium per exposure year",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(20).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap_log_cost"])
    ax.set_xlabel("Mean absolute SHAP value on log cost scale")
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_summary.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, part in monitor.groupby("scenario"):
        ax.plot(part["feature"], part["psi"], marker="o", label=label)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("PSI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "monitoring.png", dpi=150)
    plt.close(fig)

    metadata = {
        "mode": mode,
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "narwhals",
                "lightgbm",
                "shap",
                "streamlit",
            ]
        },
        "seed": SEED,
        "tweedie_power": POWER,
        "full_dataset_policies": full_rows,
        "used_policies": len(policies),
        "data_quality": data_quality,
        "train_policies": len(train),
        "test_policies": len(test),
        "positive_claim_policies_in_train": int(positive.sum()),
        "reported_claims_all": int(policies["ClaimNb"].sum()),
        "matched_claim_records_all": int(policies["ClaimRecords"].sum()),
        "matched_claim_ratio_train": float(matched_claim_ratio),
        "lightgbm_train_calibration_factor": float(calibration_factor),
        "risk_tier_cutpoints": np.quantile(train_score, [0.2, 0.4, 0.6, 0.8]).tolist(),
        "expense_ratio": 0.20,
        "risk_loading": 0.30,
        "pricing_notice": "基于公开数据与假设的模拟定价，并非实际保险报价",
        "notes": "纯风险保费及观察赔款率按暴露期计算；测试集仅用于评估。",
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_params = {
        "poisson_glm": poisson.get_params(),
        "gamma_glm": gamma.get_params(),
        "tweedie_glm": tweedie.get_params(),
        "lightgbm_tweedie": lightgbm.get_params(),
        "features": FEATURES,
        "seed": SEED,
    }
    (result_dir / "model_params.json").write_text(
        json.dumps(model_params, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
