"""Evaluate the validation-selected blend of tuned ANFIS and tree models."""

import argparse
from datetime import date
from pathlib import Path

from auto_predict.experiments.tuned_blend import run_blend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--trials-dir", type=Path, default=Path("data/processed/tuning"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/tuning/blend"))
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run_blend(args.source, args.trials_dir, args.output_dir, args.test_start, args.folds)
