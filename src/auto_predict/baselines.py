"""Train and evaluate classical price models."""

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from auto_predict.evaluation import FEATURES, TARGET, regression_metrics, temporal_folds


def fit_baselines(train: pd.DataFrame) -> dict[str, Pipeline]:
    """Fit all models, each with imputation learned from the training rows."""
    models = {
        "Median": DummyRegressor(strategy="median"),
        "Ridge": make_pipeline(StandardScaler(), Ridge()),
        "Random Forest": RandomForestRegressor(min_samples_leaf=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            loss="absolute_error", random_state=42
        ),
    }
    return {
        name: make_pipeline(SimpleImputer(strategy="median"), model).fit(
            train[FEATURES], train[TARGET]
        )
        for name, model in models.items()
    }


def evaluate_baselines(
    development: pd.DataFrame, test: pd.DataFrame, n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each fold, then refit on pre-test data and predict the holdout."""
    results = []
    for fold, (train, validation) in enumerate(temporal_folds(development, n_splits), 1):
        for name, model in fit_baselines(train).items():
            for split, subset in (("train", train), ("validation", validation)):
                predictions = model.predict(subset[FEATURES])
                results.append(_score(name, split, fold, subset, predictions))

    test_predictions = test[[TARGET]].copy()
    for name, model in fit_baselines(development).items():
        predictions = model.predict(test[FEATURES])
        test_predictions[name] = predictions
        results.append(_score(name, "test", None, test, predictions))

    metrics = pd.DataFrame(results)
    metrics["fold"] = metrics["fold"].astype("Int64")
    return metrics, test_predictions


def _score(name, split, fold, listings, predictions) -> dict:
    return {
        "model": name,
        "split": split,
        "fold": fold,
        "period_start": listings["listed_at"].min().date(),
        "period_end": listings["listed_at"].max().date(),
        "listings": len(listings),
        **regression_metrics(listings[TARGET], predictions),
    }
