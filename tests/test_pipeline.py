import pandas as pd

from used_car_price.cleaning import clean_automobile_data
from used_car_price.demo import make_demo_data
from used_car_price.modeling import train_and_compare


def test_mixed_units_are_normalised() -> None:
    raw = pd.DataFrame(
        {
            "Price": ["$10,000", "$12,000"],
            "Mileage": ["10 mi", "10 km"],
            "Horsepower": ["80 kW", "90 kW"],
            "Weight": ["2204.62 lbs", "1000 kg"],
            "Displacement": ["1400 cc", "1600 cc"],
            "Age": [2, 3],
        }
    )
    cleaned = clean_automobile_data(raw)
    assert round(cleaned.loc[0, "mileage_km"], 3) == 16.093
    assert round(cleaned.loc[0, "weight_kg"]) == 1000


def test_end_to_end_training() -> None:
    cleaned = clean_automobile_data(make_demo_data(rows=180))
    results, model = train_and_compare(cleaned)
    assert set(results) == {"linear_regression", "random_forest"}
    assert max(result.r2 for result in results.values()) > 0.5
    assert hasattr(model, "predict")

