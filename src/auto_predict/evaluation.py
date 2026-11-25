"""Shared data split and regression metrics for model comparisons."""

from collections.abc import Iterator
from datetime import date

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit


FEATURES = ["age_years", "mileage_km", "power_hp"]
TARGET = "price_rub"


def split_holdout(
    listings: pd.DataFrame, test_start: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse listing dates and reserve the last period for testing."""
    dates = pd.to_datetime(listings["listed_at"], format="ISO8601").dt.normalize()
    if dates.isna().any():
        raise ValueError("Every listing must have a date.")

    listings = listings.assign(listed_at=dates)
    before_test = dates < pd.Timestamp(test_start)
    development = listings.loc[before_test]
    test = listings.loc[~before_test]
    if development.empty or test.empty:
        raise ValueError("There must be listings both before and during the test period.")
    return development, test


def temporal_folds(
    development: pd.DataFrame, n_splits: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding training windows followed by disjoint validation periods."""
    days = development["listed_at"]
    unique_days = pd.Index(days.unique()).sort_values()
    for train_indices, validation_indices in TimeSeriesSplit(n_splits=n_splits).split(
        unique_days
    ):
        train = development.loc[days.isin(unique_days[train_indices])]
        validation = development.loc[days.isin(unique_days[validation_indices])]
        yield train, validation


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Calculate price errors in rubles."""
    return {
        "mae_rub": mean_absolute_error(actual, predicted),
        "rmse_rub": root_mean_squared_error(actual, predicted),
        "median_ae_rub": median_absolute_error(actual, predicted),
    }
