"""Tune the classical regressors on the prepared sample."""

import argparse
from datetime import date
from pathlib import Path

from auto_predict.experiments.tune_trees import run_tree_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--refine-from", type=Path, help="Refine boosting from a previous trial CSV")
    parser.add_argument("--catboost", action="store_true", help="Run the optional CatBoost experiment")
    args = parser.parse_args()
    name = "catboost" if args.catboost else "trees"
    if args.refine_from:
        name += "_refined"
    output = args.output_dir or Path("data/processed/tuning") / name
    run_tree_tuning(
        args.source, output, args.test_start, args.folds, args.refine_from, args.catboost
    )
