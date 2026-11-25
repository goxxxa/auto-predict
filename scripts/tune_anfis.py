"""Tune the GPU ANFIS with forward-chaining validation."""

import argparse
from datetime import date
from pathlib import Path

from auto_predict.experiments.tune_anfis import run_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--refine-from", type=Path, help="Check transmission using a prior trial CSV")
    parser.add_argument("--add-generation", action="store_true", help="Include vehicle generation in the refinement")
    args = parser.parse_args()
    name = "anfis"
    if args.refine_from:
        name = "anfis_vehicle" if args.add_generation else "anfis_extended"
    output = args.output_dir or Path("data/processed/tuning") / name
    run_tuning(
        args.source, output, args.test_start, args.folds,
        args.refine_from, args.add_generation,
    )
