"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .cleaning import clean_automobile_data
from .demo import make_demo_data
from .eda import create_eda_report
from .modeling import train_and_compare


def run_pipeline(raw: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = clean_automobile_data(raw)
    cleaned.to_csv(output_dir / "cleaned_auto.csv", index=False)
    create_eda_report(cleaned, output_dir / "eda")
    results, best_model = train_and_compare(cleaned)
    metrics = {name: result.to_dict() for name, result in results.items()}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(best_model, output_dir / "best_model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, explore and model used-car listing data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the complete pipeline on generated demo data")
    demo.add_argument("--output-dir", default="artifacts/demo")
    train = subparsers.add_parser("train", help="run the complete pipeline on a CSV file")
    train.add_argument("--input", required=True)
    train.add_argument("--output-dir", default="artifacts/run")
    args = parser.parse_args()

    raw = make_demo_data() if args.command == "demo" else pd.read_csv(args.input)
    metrics = run_pipeline(raw, Path(args.output_dir))
    print(json.dumps(metrics, indent=2))

