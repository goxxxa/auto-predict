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
    parser.add_argument(
        "--lookback-months", type=int, default=0, help="Training months. Use 0 for all history."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    filename = "baselines_recent.csv" if args.lookback_months else "baselines.csv"
    output = args.output or Path("data/processed") / filename
    validation_plot = output.with_name(f"{output.stem}_validation.png")
    test_plot = output.with_name(f"{output.stem}_test.png")
    outputs = output, validation_plot, test_plot
    if any(args.source.resolve() == path.resolve() for path in outputs):
        parser.error("The source and output files must have different paths.")

    try:
        listings = pd.read_csv(args.source, usecols=["listed_at", TARGET, *FEATURES])
        development, test = split_holdout(listings, args.test_start)
        metrics, predictions = evaluate_baselines(
            development, test, args.folds, args.lookback_months
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output, index=False)
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
    window = args.lookback_months or "all"
    print(f"Window: {window} months | Folds: {args.folds} | Test: {len(test):,}")
    print(summary.sort_values("validation_mae_mean").round(2).to_string())
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
