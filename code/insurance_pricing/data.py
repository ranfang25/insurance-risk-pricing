"""OpenML 数据读取及保单级关联。"""

import gzip
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FEATURES = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]


def _openml_frame(data_id: int, cache: Path, processed: Path) -> pd.DataFrame:
    local = processed / f"openml_{data_id}.csv.gz"
    if local.exists():
        return pd.read_csv(local, encoding="utf-8")
    downloaded = cache / f"openml_{data_id}.arff.gz"
    if not downloaded.exists():
        cache.mkdir(parents=True, exist_ok=True)
        metadata_url = f"https://www.openml.org/api/v1/json/data/{data_id}"
        with urlopen(metadata_url, timeout=60) as response:
            download_url = json.load(response)["data_set_description"]["url"]
        request = Request(download_url, headers={"Accept-Encoding": "gzip"})
        temporary = cache / f"openml_{data_id}.part"
        with urlopen(request, timeout=120) as response:
            if response.headers.get("Content-Encoding", "").lower() == "gzip":
                with temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            else:
                with gzip.open(temporary, "wb") as target:
                    shutil.copyfileobj(response, target)
        temporary.replace(downloaded)
    with gzip.open(downloaded, "rt", encoding="utf-8") as stream:
        columns = []
        for line in stream:
            stripped = line.strip()
            if stripped.lower().startswith("@attribute"):
                columns.append(stripped.split(maxsplit=2)[1])
            if stripped.lower() == "@data":
                break
        frame = pd.read_csv(stream, header=None, names=columns, quotechar="'")
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(local, index=False, encoding="utf-8", compression="gzip")
    return frame


def join_policies(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """先汇总同一保单的多笔赔款，再做一对一关联。"""
    required_freq = {"IDpol", "ClaimNb", "Exposure", *FEATURES}
    required_sev = {"IDpol", "ClaimAmount"}
    if missing := required_freq.difference(freq.columns):
        raise ValueError(f"频率数据缺少字段: {sorted(missing)}")
    if missing := required_sev.difference(sev.columns):
        raise ValueError(f"赔款数据缺少字段: {sorted(missing)}")

    policies = freq.copy()
    policies["IDpol"] = pd.to_numeric(policies["IDpol"], errors="raise").astype("int64")
    if policies["IDpol"].duplicated().any():
        raise ValueError("频率数据中的 IDpol 非唯一")
    claims = sev[["IDpol", "ClaimAmount"]].copy()
    claims["IDpol"] = pd.to_numeric(claims["IDpol"], errors="raise").astype("int64")
    claims["ClaimAmount"] = pd.to_numeric(claims["ClaimAmount"], errors="coerce")
    claims = claims.dropna(subset=["ClaimAmount"])
    if (claims["ClaimAmount"] < 0).any():
        raise ValueError("赔款数据存在负金额")
    sums = claims.groupby("IDpol", as_index=False).agg(
        ClaimAmount=("ClaimAmount", "sum"), ClaimRecords=("ClaimAmount", "size")
    )
    policies = policies.merge(sums, on="IDpol", how="left", validate="one_to_one")
    policies[["ClaimAmount", "ClaimRecords"]] = policies[["ClaimAmount", "ClaimRecords"]].fillna(0)
    raw_policies = len(policies)
    for column in ["ClaimNb", "Exposure", *NUMERIC]:
        policies[column] = pd.to_numeric(policies[column], errors="coerce")
    policies = policies.dropna(subset=["ClaimNb", "Exposure", *NUMERIC])
    valid = (
        (policies["Exposure"] > 0)
        & (policies["ClaimNb"] >= 0)
        & (policies["VehPower"] > 0)
        & (policies["VehAge"] >= 0)
        & (policies["DrivAge"] >= 18)
        & (policies["BonusMalus"] > 0)
        & (policies["Density"] >= 0)
    )
    policies = policies[valid].copy()
    for column in CATEGORICAL:
        policies[column] = policies[column].astype("string").fillna("Unknown").astype(str)
    policies["Exposure"] = policies["Exposure"].clip(upper=1)
    policies["ClaimNb"] = policies["ClaimNb"].astype(int)
    policies["ClaimRecords"] = policies["ClaimRecords"].astype(int)
    policies["Frequency"] = policies["ClaimNb"] / policies["Exposure"]
    policies["Severity"] = np.where(
        policies["ClaimRecords"] > 0,
        policies["ClaimAmount"] / policies["ClaimRecords"].clip(lower=1),
        0,
    )
    policies["PurePremium"] = policies["ClaimAmount"] / policies["Exposure"]
    policies.attrs["quality"] = {
        "raw_policies": raw_policies,
        "removed_missing_or_invalid": raw_policies - len(policies),
        "source_claim_records": len(claims),
        "claim_records_without_valid_policy": len(claims) - int(policies["ClaimRecords"].sum()),
    }
    return policies


def load_policies(cache: Path) -> pd.DataFrame:
    processed = cache.parent / "processed"
    return join_policies(
        _openml_frame(41214, cache, processed),
        _openml_frame(41215, cache, processed),
    )
