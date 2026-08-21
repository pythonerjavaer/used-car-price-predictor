"""Deterministic synthetic data for a zero-setup demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(rows: int = 320, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 14, rows)
    mileage = np.maximum(500, age * rng.normal(14500, 3200, rows) + rng.normal(9000, 7000, rows))
    horsepower = rng.normal(105, 30, rows).clip(45, 260)
    weight = rng.normal(1350, 230, rows).clip(850, 2400)
    displacement = (horsepower * rng.normal(14, 2, rows)).clip(900, 4200)
    fuel = rng.choice(["Benzine", "Diesel", "Hybrid"], rows, p=[0.5, 0.36, 0.14])
    body = rng.choice(["Sedans", "Station wagon", "SUV", "Compact"], rows)
    price = (
        15500
        - mileage * 0.055
        - age * 430
        + horsepower * 92
        + (body == "SUV") * 3200
        + (fuel == "Hybrid") * 2100
        + rng.normal(0, 1800, rows)
    ).clip(2500, 65000)
    return pd.DataFrame(
        {
            "Make_Model": rng.choice(["Aster A1", "Boreal X", "Cobalt S", "Delta Tourer"], rows),
            "Body_Type": body,
            "Price": [f"${value:,.0f}" for value in price],
            "Mileage": [f"{value / 1.609344:.1f} mi" if i % 4 == 0 else f"{value:.1f} km" for i, value in enumerate(mileage)],
            "Type": "Used",
            "Fuel": fuel,
            "Gears": rng.choice([5, 6, 7], rows),
            "Age": age,
            "Previous_Owners": rng.integers(1, 4, rows),
            "Horsepower": [f"{value:.1f} kW" for value in horsepower],
            "Gearing_Type": rng.choice(["Automatic", "Manual"], rows),
            "Displacement": [f"{value:.0f} cc" for value in displacement],
            "Weight": [f"{value / 0.45359237:.1f} lbs" if i % 5 == 0 else f"{value:.1f} kg" for i, value in enumerate(weight)],
            "Drive_Chain": rng.choice(["front", "rear", "4WD"], rows),
            "Cons_Comb": rng.normal(6.2, 1.1, rows).clip(3, 12),
        }
    )

