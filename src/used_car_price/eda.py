"""Exploratory analysis helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_eda_report(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create a compact set of portfolio-ready EDA charts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs: list[Path] = []

    charts = [
        ("price_distribution.png", lambda ax: sns.histplot(df["price"], kde=True, ax=ax)),
        (
            "price_vs_mileage.png",
            lambda ax: sns.scatterplot(data=df, x="mileage_km", y="price", hue="fuel", alpha=0.65, ax=ax),
        ),
        (
            "price_by_body_type.png",
            lambda ax: sns.boxplot(data=df, x="body_type", y="price", ax=ax),
        ),
        (
            "numeric_correlations.png",
            lambda ax: sns.heatmap(
                df[["price", "mileage_km", "horsepower_kw", "weight_kg", "displacement_cc", "age"]].corr(),
                cmap="vlag",
                center=0,
                annot=True,
                fmt=".2f",
                ax=ax,
            ),
        ),
    ]

    for filename, draw in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax)
        ax.set_title(filename.removesuffix(".png").replace("_", " ").title())
        if filename == "price_by_body_type.png":
            ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = target / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    return outputs

