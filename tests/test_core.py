"""核心计算与数据关联的最小回归测试。"""

import numpy as np
import pandas as pd
from insurance_pricing.data import join_policies
from insurance_pricing.metrics import assign_tiers, normalized_gini
from insurance_pricing.monitoring import psi


def test_join_aggregates_multiple_claims() -> None:
    freq = pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [2, 0],
            "Exposure": [0.5, 1.0],
            "Area": ["A", "B"],
            "VehPower": [5, 6],
            "VehAge": [3, 4],
            "DrivAge": [30, 40],
            "BonusMalus": [50, 50],
            "VehBrand": ["B1", "B2"],
            "VehGas": ["Diesel", "Regular"],
            "Density": [100, 200],
            "Region": ["R1", "R2"],
        }
    )
    sev = pd.DataFrame({"IDpol": [1, 1], "ClaimAmount": [100, 200]})
    result = join_policies(freq, sev)
    assert result.loc[0, "ClaimAmount"] == 300
    assert result.loc[0, "ClaimRecords"] == 2
    assert result.loc[0, "PurePremium"] == 600
    assert result.loc[1, "ClaimAmount"] == 0


def test_gini_ranks_perfect_above_reverse() -> None:
    actual = np.array([0, 0, 1, 4], dtype=float)
    weight = np.ones(4)
    assert np.isclose(normalized_gini(actual, actual, weight), 1)
    assert normalized_gini(actual, actual[::-1], weight) < 0


def test_psi_identical_distribution_is_zero() -> None:
    ref = np.arange(100, dtype=float)
    assert np.isclose(psi(ref, ref.copy()), 0)
    assert psi(ref, ref + 30) > 0


def test_tiers_use_reference_cutpoints() -> None:
    ref = pd.Series(np.arange(1, 101))
    actual = pd.Series([1, 20, 50, 80, 100])
    assert assign_tiers(ref, actual).tolist() == [1, 1, 3, 4, 5]
