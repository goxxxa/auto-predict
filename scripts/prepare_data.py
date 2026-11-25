"""Prepare the working sample from a Drom CSV."""

import argparse
from pathlib import Path

from auto_predict.data import prepare_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare listings from a Drom CSV.")
    parser.add_argument("source", type=Path, help="Path to the downloaded Drom CSV")
    parser.add_argument("--model", default="Kia Rio")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--vehicle-details", action="store_true", help="Include extra vehicle and geographic features")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    default_output = "data/processed/extended" if args.vehicle_details else "data/processed"
    output_dir = args.output_dir or Path(default_output)
    listings_path = output_dir / "listings.csv"
    report_path = output_dir / "quality_by_year.csv"
    if args.source.resolve() in (listings_path.resolve(), report_path.resolve()):
        parser.error("The source CSV and output files must have different paths.")

    try:
        sample, report = prepare_sample(
            args.source, args.model, args.start_year, args.end_year, args.chunk_size,
            include_vehicle_details=args.vehicle_details,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(listings_path, index=False)
        report.to_csv(report_path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    print(report.to_string())
    print(f"Saved {len(sample):,} listings with valid prices to {output_dir}")


if __name__ == "__main__":
    main()
