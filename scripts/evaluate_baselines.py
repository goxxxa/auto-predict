"""Run the baseline comparison on the prepared Drom listings."""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from auto_predict.evaluation import FEATURES, TARGET, split_holdout
from auto_predict.experiments.baselines import evaluate_baselines
from auto_predict.reports.plots import plot_fold_errors, plot_test_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the prepared listings CSV")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    parser.add_argument("--output", type=Path, default=Path("data/processed/baselines.csv"))
    args = parser.parse_args()

    validation_plot = args.output.with_name(f"{args.output.stem}_validation.png")
    test_plot = args.output.with_name(f"{args.output.stem}_test.png")
    outputs = args.output, validation_plot, test_plot
    if any(args.source.resolve() == output.resolve() for output in outputs):
        parser.error("The source and output files must have different paths.")

    try:
        listings = pd.read_csv(args.source, usecols=["listed_at", TARGET, *FEATURES])
        development, test = split_holdout(listings, args.test_start)
        metrics, predictions = evaluate_baselines(development, test, args.folds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output, index=False)
        plot_fold_errors(metrics, validation_plot)
        plot_test_predictions(predictions, metrics, test_plot)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    validation_mae = metrics.loc[metrics["split"] == "validation"].groupby("model")[
        "mae_rub"
    ]
    summary = validation_mae.agg(validation_mae_mean="mean", validation_mae_std="std")
    test_mae = metrics.loc[metrics["split"] == "test"].set_index("model")["mae_rub"]
    summary["test_mae"] = test_mae
    print(f"Pre-test: {len(development):,} | Folds: {args.folds} | Test: {len(test):,}")
    print(summary.sort_values("validation_mae_mean").round(2).to_string())
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
