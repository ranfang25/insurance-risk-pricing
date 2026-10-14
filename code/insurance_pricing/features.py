"""模型特征转换。"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insurance_pricing.data import CATEGORICAL, NUMERIC


def make_encoder() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
        ],
        sparse_threshold=1,
    )


def clean_predictions(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 1e-9)
