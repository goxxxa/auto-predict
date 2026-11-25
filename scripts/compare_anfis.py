"""Compare ANFIS and classical models and export rule explanations."""

import argparse
from datetime import date
from pathlib import Path

from auto_predict.anfis_report import write_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the prepared listings CSV")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/anfis_comparison"))
    parser.add_argument("--test-start", type=date.fromisoformat, default="2025-01-01")
    args = parser.parse_args()
    try:
        write_comparison(args.source, args.runs_dir, args.output_dir, args.test_start)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
