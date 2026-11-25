"""Tune classical tree regressors on common forward-chaining folds."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from auto_predict.evaluation import (
    FEATURES, TARGET, regression_metrics, split_holdout, temporal_folds, training_window_days,
)
from auto_predict.reports.plots import plot_test_predictions


WINDOWS = [10, 14, 21, 30, 45, 60, 90, 120]
VEHICLE = ["engine_liters", "generation", "facelift", "transmission", "body_type"]
GEOGRAPHY = ["region", "macro_region"]
CATEGORIES = ["transmission", "body_type", *GEOGRAPHY]
HISTOGRAM = "Histogram Gradient Boosting"
CATBOOST = "CatBoost"


def predict_tree(train, validation, features, config):
    model_name = config["model"]
    controls = {"model", "feature_set", "days", "log_target", "age_basis",
                "half_life_days", "calibration_days"}
    options = {key: value for key, value in config.items() if key not in controls}
    calibration = None
    if config.get("calibration_days"):
        cutoff = train["listed_at"].max() - pd.Timedelta(days=config["calibration_days"] - 1)
        earlier = train["listed_at"].lt(cutoff)
        if earlier.any():
            calibration = train.loc[~earlier]
            train = train.loc[earlier]

    def inputs(listings):
        values = listings[features]
        if config.get("age_basis") == "cohort":
            values = values.assign(age_years=2000 - listings["model_year"])
        if model_name == CATBOOST:
            categories = [column for column in CATEGORIES if column in features]
            values = values.assign(**{
                column: values[column].astype("string").fillna("missing") for column in categories
            })
        return values

    target = train[TARGET] / 1_000_000
    if config["log_target"]:
        target = np.log(target)
    if model_name == HISTOGRAM:
        model = HistGradientBoostingRegressor(**options, early_stopping=False, random_state=42)
    elif model_name == CATBOOST:
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("Use uv run --extra boost for CatBoost tuning.") from error

        model = CatBoostRegressor(
            **options, cat_features=[column for column in CATEGORIES if column in features],
            thread_count=2, verbose=False, allow_writing_files=False, random_seed=42,
        )
    else:
        estimator = GradientBoostingRegressor if model_name == "Gradient Boosting" else RandomForestRegressor
        model = make_pipeline(SimpleImputer(strategy="median"), estimator(**options, random_state=42))
    fit_options = {}
    if config.get("half_life_days"):
        age = (train["listed_at"].max() - train["listed_at"]).dt.days
        fit_options["sample_weight"] = np.exp2(-age / config["half_life_days"])
    model.fit(inputs(train), target, **fit_options)

    def estimate(listings):
        prediction = model.predict(inputs(listings))
        if config["log_target"]:
            prediction = np.exp(prediction)
        return prediction * 1_000_000

    prediction = estimate(validation)
    if calibration is not None:
        prediction += np.median(calibration[TARGET] - estimate(calibration))
    return prediction


def tree_feature_sets(listings: pd.DataFrame) -> dict[str, list[str]]:
    feature_sets = {"basic": FEATURES}
    if set(VEHICLE).issubset(listings):
        feature_sets["vehicle"] = [*FEATURES, *VEHICLE]
    if set([*VEHICLE, *GEOGRAPHY]).issubset(listings):
        feature_sets["vehicle and region"] = [*FEATURES, *VEHICLE, *GEOGRAPHY]
    return feature_sets


def tree_config_from_row(row: pd.Series, model: str) -> dict:
    feature_set = row.get("feature_set")
    if feature_set is None:
        feature_set = "vehicle and region" if row["geography"] else "vehicle"
    config = {
        "model": model, "feature_set": feature_set,
        "days": int(row["days"]), "log_target": bool(row["log_target"]),
        "learning_rate": float(row["learning_rate"]),
    }
    if model == HISTOGRAM:
        config.update(
            loss=row["loss"], max_leaf_nodes=int(row["max_leaf_nodes"]),
            min_samples_leaf=int(row["min_samples_leaf"]),
            l2_regularization=float(row["l2_regularization"]), max_iter=int(row["max_iter"]),
        )
        if row["loss"] == "quantile":
            config["quantile"] = float(row["quantile"])
    else:
        config.update(
            loss_function=row["loss_function"], depth=int(row["depth"]),
            iterations=int(row["iterations"]), l2_leaf_reg=float(row["l2_leaf_reg"]),
            half_life_days=int(row.get("half_life_days", row.get("half_life", 0))),
        )
    if pd.notna(row.get("age_basis")):
        config["age_basis"] = row["age_basis"]
    for option in ("half_life_days", "calibration_days"):
        if pd.notna(row.get(option)):
            config[option] = int(row[option])
    return config


def _histogram_refinements(reference: Path):
    tested = pd.read_csv(reference)
    histogram = tested.loc[tested["model"].eq(HISTOGRAM)]
    for _, rows in histogram.groupby("feature_set"):
        for _, row in rows.nsmallest(2, "validation_mae_rub").iterrows():
            config = tree_config_from_row(row, HISTOGRAM)
            yield config
            yield {**config, "age_basis": "cohort"}
            for half_life in (5, 10, 21):
                yield {**config, "age_basis": "cohort", "half_life_days": half_life,
                       "days": max(60, config["days"])}
            for span in (45, 90):
                yield {**config, "age_basis": "cohort", "calibration_days": 7, "days": span}


def _catboost_candidates(reference: Path | None):
    if reference is None:
        grid = {
            "days": [14, 21, 30, 45, 60, 90], "half_life": [0, 7, 21],
            "geography": [False, True], "log_target": [False, True],
            "loss_function": ["MAE", "RMSE"], "depth": [4, 5, 6],
            "iterations": [160, 260], "learning_rate": [0.04, 0.075, 0.12],
            "l2_leaf_reg": [1, 4, 12],
        }
        for sampled in ParameterSampler(grid, n_iter=36, random_state=42):
            sampled["model"] = CATBOOST
            sampled["feature_set"] = "vehicle and region" if sampled.pop("geography") else "vehicle"
            sampled["half_life_days"] = sampled.pop("half_life")
            yield sampled
        return

    row = pd.read_csv(reference).nsmallest(1, "validation_mae_rub").iloc[0]
    config = tree_config_from_row(row, CATBOOST)
    yield config
    cohort = {**config, "age_basis": "cohort"}
    yield cohort
    for days in (30, 60, 90):
        for half_life in (7, 14, 30):
            yield {**cohort, "days": days, "half_life_days": half_life}
    for days in (45, 90):
        yield {**cohort, "days": days, "calibration_days": 7}
    for quantile in (0.55, 0.65):
        yield {**cohort, "loss_function": f"Quantile:alpha={quantile}"}


def run_tree_tuning(
    source: Path, output: Path, test_start: date, folds: int, refine_from: Path | None = None,
    catboost: bool = False,
) -> None:
    listings = pd.read_csv(source)
    for column in CATEGORIES:
        if column in listings:
            listings[column] = listings[column].astype("category")
    feature_sets = tree_feature_sets(listings)
    development, test = split_holdout(listings, test_start)
    splits = list(temporal_folds(development, folds))
    results = []
    checked = set()
    best = {}

    def evaluate(config: dict) -> None:
        key = tuple(sorted(config.items()))
        if key in checked:
            return
        checked.add(key)
        features = feature_sets[config["feature_set"]]
        errors = []
        for train, validation in splits:
            recent = training_window_days(train, config["days"])
            prediction = predict_tree(recent, validation, features, config)
            errors.append((validation[TARGET] - prediction).abs().mean())
        score = sum(errors) / len(errors)
        row = {
            **config, "validation_mae_rub": score,
            **{f"fold_{i}_mae_rub": value for i, value in enumerate(errors, 1)},
        }
        results.append(row)
        group = (config["model"], config["feature_set"])
        if group not in best or score < best[group][0]:
            best[group] = score, config
        if len(results) % 10 == 0:
            print(f"Tree trial {len(results)}: best MAE {min(item[0] for item in best.values()):,.0f} RUB", flush=True)

    histogram_grid = {
        "days": [14, 21, 30, 45, 60, 90, 120],
        "loss": ["absolute_error", "squared_error"],
        "log_target": [False, True],
        "max_leaf_nodes": [7, 15, 31, 63],
        "min_samples_leaf": [8, 16, 30, 60],
        "l2_regularization": [0, 0.1, 1, 5, 20],
        "learning_rate": [0.025, 0.05, 0.1],
        "max_iter": [80, 160, 300],
    }
    classic = {
        "Gradient Boosting": {
            "loss": ["absolute_error", "huber", "squared_error"],
            "n_estimators": [80, 160, 260],
            "learning_rate": [0.03, 0.07, 0.1],
            "max_depth": [2, 3, 5],
            "min_samples_leaf": [8, 20, 45],
            "subsample": [0.7, 0.9, 1.0],
        },
        "Random Forest": {
            "n_estimators": [60, 120],
            "max_depth": [5, 8, 14, None],
            "min_samples_leaf": [4, 10, 20, 40],
            "max_features": [0.7, 1.0],
        },
    }

    with threadpool_limits(limits=2):
        if catboost:
            if "vehicle and region" not in feature_sets:
                raise ValueError("CatBoost tuning needs the extended CSV.")
            for config in _catboost_candidates(refine_from):
                evaluate(config)
        elif refine_from is None:
            for feature_set in feature_sets:
                for options in ParameterSampler(histogram_grid, n_iter=36, random_state=42):
                    evaluate({"model": HISTOGRAM, "feature_set": feature_set, **options})
                _, leader = best[HISTOGRAM, feature_set]
                for days in (7, 10, 21, 35, 75):
                    evaluate({**leader, "days": days})
                for quantile in (0.35, 0.45, 0.55, 0.65, 0.75):
                    evaluate({**leader, "loss": "quantile", "quantile": quantile})

            for model, settings in classic.items():
                grid = {"days": WINDOWS, "log_target": [False, True], **settings}
                for options in ParameterSampler(grid, n_iter=24, random_state=42):
                    evaluate({"model": model, "feature_set": "basic", **options})
        else:
            for config in _histogram_refinements(refine_from):
                evaluate(config)

        predictions = test[[TARGET]].copy()
        summaries = []
        for (model, feature_set), (score, config) in best.items():
            recent = training_window_days(development, config["days"])
            predicted = predict_tree(recent, test, feature_sets[feature_set], config)
            name = f"{model} ({feature_set})"
            predictions[name] = predicted
            relative = (test[TARGET] - predicted).abs() / test[TARGET]
            summaries.append({
                "model": name, "split": "test", "days": config["days"],
                "validation_mae_rub": score, "test_listings": len(test),
                **regression_metrics(test[TARGET], predicted),
                "mape_percent": 100 * relative.mean(),
                "within_20_percent": 100 * relative.le(0.2).mean(),
            })

    ranking = pd.DataFrame(results).sort_values("validation_mae_rub")
    summary = pd.DataFrame(summaries).sort_values("validation_mae_rub")
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / "validation_trials.csv", index=False)
    summary.to_csv(output / "test_metrics.csv", index=False)
    plot_test_predictions(predictions, summary, output / "test.png")
    print(summary.round(2).to_string(index=False))
