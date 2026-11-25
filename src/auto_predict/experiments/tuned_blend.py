"""Test a convex blend of the validation-selected fuzzy and tree models."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from auto_predict.evaluation import (
    TARGET, regression_metrics, split_holdout, temporal_folds, training_window_days,
)
from auto_predict.experiments.anfis_genetic import predict_prices
from auto_predict.experiments.tune_anfis import encode_features, fit_trial, trial_from_row
from auto_predict.experiments.tune_trees import (
    CATBOOST, CATEGORIES, HISTOGRAM, predict_tree, tree_config_from_row, tree_feature_sets,
)
from auto_predict.reports.plots import plot_model_comparison, plot_test_predictions


def _best_row(path: Path) -> pd.Series:
    return pd.read_csv(path).nsmallest(1, "validation_mae_rub").iloc[0]


def run_blend(source: Path, trials_dir: Path, output: Path, test_start: date, folds: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train the ANFIS component.")
    device = torch.device("cuda:0")
    listings = pd.read_csv(source)
    for column in CATEGORIES:
        listings[column] = listings[column].astype("category")
    features = tree_feature_sets(listings)
    development, test = split_holdout(listings, test_start)
    trial = trial_from_row(_best_row(trials_dir / "anfis_vehicle/validation_trials.csv"))
    histogram = tree_config_from_row(
        _best_row(trials_dir / "trees_refined/validation_trials.csv"), HISTOGRAM
    )
    catboost = tree_config_from_row(
        _best_row(trials_dir / "catboost_refined/validation_trials.csv"), CATBOOST
    )
    model_names = ["Tuned ANFIS + GA", HISTOGRAM, CATBOOST]

    def predictions(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
        model, scaler, _, _ = fit_trial(encode_features(train, trial, test_start.year), trial, device)
        fuzzy = predict_prices(
            model, encode_features(evaluation, trial, test_start.year), scaler,
            log_target=trial.fit.log_target,
        )
        trees = [
            predict_tree(training_window_days(train, config["days"]), evaluation,
                         features[config["feature_set"]], config)
            for config in (histogram, catboost)
        ]
        return np.column_stack([fuzzy.to_numpy(), *trees])

    candidates = np.array(
        [(a, b, 10 - a - b) for a in range(11) for b in range(11 - a)], dtype=float
    ) / 10
    with threadpool_limits(limits=2):
        errors = []
        individual_mae = []
        for index, (train, validation) in enumerate(temporal_folds(development, folds), 1):
            predicted = predictions(train, validation)
            actual = validation[TARGET].to_numpy()[:, None]
            errors.append(np.abs(predicted @ candidates.T - actual).mean(axis=0))
            individual_mae.append(np.abs(predicted - actual).mean(axis=0))
            print(f"Blend fold {index}/{folds}", flush=True)
        errors = np.asarray(errors)
        mean_mae = errors.mean(axis=0)
        selected = int(mean_mae.argmin())
        weights = candidates[selected]
        test_predictions = predictions(development, test)

    names = [*model_names, "Blend"]
    validation_mae = np.column_stack((individual_mae, errors[:, selected]))
    test_predictions = np.column_stack((test_predictions, test_predictions @ weights))
    actual = test[TARGET].to_numpy()
    rows = []
    expensive = actual >= 1_500_000
    for name, fold_mae, predicted in zip(names, validation_mae.T, test_predictions.T):
        relative = np.abs(actual - predicted) / actual
        rows.append({
            "model": name, "split": "test", "validation_mae_rub": fold_mae.mean(),
            **regression_metrics(actual, predicted), "mape_percent": 100 * relative.mean(),
            "within_20_percent": 100 * np.mean(relative <= 0.2),
            "mean_error_rub": (predicted - actual).mean(),
            "expensive_mae_rub": np.abs(predicted[expensive] - actual[expensive]).mean()
            if expensive.any() else np.nan,
        })
    metrics = pd.DataFrame(rows).sort_values("validation_mae_rub")
    ranking = pd.DataFrame(candidates, columns=model_names).assign(validation_mae_rub=mean_mae)
    fold_metrics = pd.DataFrame(validation_mae, columns=names)
    fold_metrics.index = pd.RangeIndex(1, folds + 1, name="fold")
    fold_metrics = fold_metrics.reset_index().melt(
        id_vars="fold", var_name="model", value_name="mae_rub"
    )

    output.mkdir(parents=True, exist_ok=True)
    ranking.sort_values("validation_mae_rub").to_csv(output / "validation_weights.csv", index=False)
    metrics.to_csv(output / "comparison.csv", index=False)
    plot_model_comparison(
        fold_metrics, metrics.sort_values("mae_rub").set_index("model").rename(
            columns={"mae_rub": "test_mae_rub"}
        ), output / "comparison.png",
    )
    points = pd.DataFrame(test_predictions, columns=names, index=test.index)
    plot_test_predictions(test[[TARGET]].join(points), metrics, output / "test.png")
    print("Blend weights:", dict(zip(model_names, weights.tolist())))
    print(metrics.round(2).to_string(index=False))
