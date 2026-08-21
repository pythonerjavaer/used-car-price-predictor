"""Data cleaning and feature engineering for used-car listings."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


RENAME_MAP = {
    "Make_Model": "make_model",
    "Body_Type": "body_type",
    "Price": "price",
    "Vat": "vat",
    "Mileage": "mileage_km",
    "Type": "type",
    "Fuel": "fuel",
    "Gears": "gears",
    "Age": "age",
    "Previous_Owners": "previous_owners",
    "Horsepower": "horsepower_kw",
    "Inspection_New": "inspection_new",
    "Paint_Type": "paint_type",
    "Gearing_Type": "gearing_type",
    "Displacement": "displacement_cc",
    "Weight": "weight_kg",
    "Drive_Chain": "drive_chain",
    "Cons_Comb": "cons_comb",
}


def _number(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def _mileage_km(value: object) -> float:
    number = _number(value)
    if np.isnan(number):
        return number
    text = str(value).lower()
    return number * 1.609344 if "mi" in text and "km" not in text else number


def _weight_kg(value: object) -> float:
    number = _number(value)
    if np.isnan(number):
        return number
    return number * 0.45359237 if "lb" in str(value).lower() else number


def clean_automobile_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready, consistently typed automobile dataset.

    The function accepts either the original title-cased columns or already
    normalised snake-case columns. It converts currency and mixed units,
    imputes practical defaults, clips implausible extremes and derives two
    interpretable efficiency features.
    """

    df = raw.rename(columns=RENAME_MAP).copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(column).strip().lower() for column in df.columns]

    required = {"price", "mileage_km", "horsepower_kw", "weight_kg", "displacement_cc"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["price"] = df["price"].map(_number)
    df["mileage_km"] = df["mileage_km"].map(_mileage_km)
    df["horsepower_kw"] = df["horsepower_kw"].map(_number)
    df["weight_kg"] = df["weight_kg"].map(_weight_kg)
    df["displacement_cc"] = df["displacement_cc"].map(_number)

    numeric_columns = [
        "gears",
        "age",
        "previous_owners",
        "inspection_new",
        "cons_comb",
    ]
    for column in numeric_columns:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates()
    df = df[df["price"].gt(0)].copy()

    for column in [
        "mileage_km",
        "horsepower_kw",
        "weight_kg",
        "displacement_cc",
        *numeric_columns,
    ]:
        if df[column].notna().any():
            df[column] = df[column].fillna(df[column].median())

    for column in ["make_model", "body_type", "type", "fuel", "gearing_type", "drive_chain"]:
        if column not in df:
            df[column] = "Unknown"
        df[column] = df[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    if len(df) >= 20:
        for column in ["price", "mileage_km", "horsepower_kw", "weight_kg", "displacement_cc"]:
            lower, upper = df[column].quantile([0.01, 0.99])
            df[column] = df[column].clip(lower=lower, upper=upper)

    df["mileage_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    df["power_to_weight"] = df["horsepower_kw"] / df["weight_kg"].replace(0, np.nan)
    return df.reset_index(drop=True)
