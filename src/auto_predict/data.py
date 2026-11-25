"""Build a working sample and annual quality report from the Drom archive."""

from pathlib import Path

import numpy as np
import pandas as pd


DROM_COLUMN_NAMES = {
    "Название машины": "model",
    "Ссылка": "listing_url",
    "Дата размещения объявления": "listed_at",
    "Год": "model_year",
    "Цена": "price_rub",
    "Пробег": "mileage_km",
    "Мощность": "power_hp",
}

DROM_VEHICLE_COLUMNS = {
    "Объем двигателя": "engine_liters",
    "Коробка передач": "transmission",
    "Поколение": "generation",
    "Рестайлинг": "facelift",
    "Тип кузова": "body_type",
    "Регион": "region",
    "Макро-регион": "macro_region",
}


def prepare_sample(
    source: Path,
    model: str,
    start_year: int,
    end_year: int,
    chunk_size: int,
    *,
    include_vehicle_details: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned listings and a quality report for the requested years."""
    model = " ".join(model.split()).casefold()
    if not model or not 1886 <= start_year <= end_year <= 9999 or chunk_size <= 0:
        raise ValueError("Specify a model, a valid year range, and a positive chunk size.")

    columns = DROM_COLUMN_NAMES
    if include_vehicle_details:
        columns = columns | DROM_VEHICLE_COLUMNS

    selected = []
    with pd.read_csv(
        source, usecols=list(columns), dtype="string", chunksize=chunk_size
    ) as chunks:
        for chunk in chunks:
            chunk = chunk.rename(columns=columns)
            names = chunk["model"].str.split().str.join(" ").str.casefold()
            segment = chunk.loc[names.eq(model)].copy()
            dates = pd.to_datetime(segment["listed_at"], format="mixed", errors="coerce")
            in_period = dates.dt.year.between(start_year, end_year)
            segment = segment.loc[in_period].copy()
            segment["listed_at"] = dates.loc[in_period]
            if not segment.empty:
                selected.append(segment)

    if not selected:
        raise ValueError("No listings match the requested model and year range.")

    sample = pd.concat(selected, ignore_index=True)
    sample["listing_url"] = sample["listing_url"].str.strip().replace("", pd.NA)
    has_url = sample["listing_url"].notna()
    sample = sample.loc[~(has_url & sample["listing_url"].duplicated())].copy()
    years = sample["listed_at"].dt.year

    numeric_columns = ["model_year", "price_rub", "mileage_km", "power_hp"]
    if include_vehicle_details:
        numeric_columns.extend(["engine_liters", "generation", "facelift"])
    for column in numeric_columns:
        values = pd.to_numeric(sample[column], errors="coerce")
        sample[column] = values.where(np.isfinite(values))

    year = sample["model_year"]
    sample["model_year"] = year.where(year.between(1886, years) & year.mod(1).eq(0))
    sample["price_rub"] = sample["price_rub"].where(sample["price_rub"].gt(0))
    sample["mileage_km"] = sample["mileage_km"].where(sample["mileage_km"].ge(0))
    sample["power_hp"] = sample["power_hp"].where(sample["power_hp"].gt(0))
    sample["age_years"] = years - sample["model_year"]

    # Keep rows with missing features for the quality report and later preprocessing.
    report = sample.groupby(years.rename("listing_year")).agg(
        listings=("price_rub", "size"),
        valid_prices=("price_rub", "count"),
        median_price_rub=("price_rub", "median"),
        known_age=("age_years", "count"),
        known_mileage=("mileage_km", "count"),
        known_power=("power_hp", "count"),
    )
    report = report.reindex(range(start_year, end_year + 1))
    counts = ["listings", "valid_prices", "known_age", "known_mileage", "known_power"]
    report[counts] = report[counts].fillna(0).astype(int)
    for feature in ("age", "mileage", "power"):
        report[f"missing_{feature}"] = report["listings"] - report.pop(f"known_{feature}")
    valid_listings = sample.loc[sample["price_rub"].notna()].sort_values("listed_at")
    return valid_listings, report
