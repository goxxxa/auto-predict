"""Train and evaluate the first-order Sugeno ANFIS on CUDA."""

import argparse
from datetime import date
from pathlib import Path

from auto_predict.anfis_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the prepared listings CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/anfis"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    args = parser.parse_args()

    try:
        run_experiment(args.source, args.output_dir, args.epochs, args.folds, args.test_start)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
