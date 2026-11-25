"""Run the CUDA ANFIS genetic experiment."""

import argparse
from datetime import date
from pathlib import Path

from auto_predict.experiments.anfis_genetic import run_genetic_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the prepared listings CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/anfis_genetic"))
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    args = parser.parse_args()

    try:
        run_genetic_experiment(
            args.source, args.output_dir, args.population, args.generations, args.folds, args.test_start
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
