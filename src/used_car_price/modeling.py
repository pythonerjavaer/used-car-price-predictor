"""Model training and comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "mileage_km",
    "horsepower_kw",
    "weight_kg",
    "displacement_cc",
    "age",
    "previous_owners",
    "gears",
    "cons_comb",
    "mileage_per_year",
    "power_to_weight",
]
CATEGORICAL_FEATURES = ["fuel", "body_type", "drive_chain", "gearing_type", "type"]


@dataclass(frozen=True)
class ModelResult:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def train_and_compare(
    df: pd.DataFrame, *, random_state: int = 42
) -> tuple[dict[str, ModelResult], Pipeline]:
    """Train a linear baseline and random forest, returning the best pipeline."""

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = sorted(set(features + ["price"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing model columns: {', '.join(missing)}")
    if len(df) < 20:
        raise ValueError("At least 20 cleaned rows are required for evaluation")

    X = df[features]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    estimators = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    results: dict[str, ModelResult] = {}
    pipelines: dict[str, Pipeline] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("prepare", _preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[name] = ModelResult(
            mae=float(mean_absolute_error(y_test, predictions)),
            rmse=float(mean_squared_error(y_test, predictions) ** 0.5),
            r2=float(r2_score(y_test, predictions)),
        )
        pipelines[name] = pipeline

    best_name = max(results, key=lambda name: results[name].r2)
    return results, pipelines[best_name]

