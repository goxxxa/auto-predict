"""Evaluate genetic ANFIS with an inner search in each temporal split."""

from datetime import date
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from auto_predict.evaluation import (
    FEATURES,
    TARGET,
    regression_metrics,
    split_holdout,
    temporal_folds,
)
from auto_predict.experiments.anfis import (
    PRICE_SCALE,
    fit_preprocessor,
    listing_tensors,
    save_checkpoint,
)
from auto_predict.models.anfis import ANFIS
from auto_predict.models.anfis_genetic import search_memberships
from auto_predict.reports.plots import plot_genetic_validation, plot_test_predictions


SEEDS = (17, 42, 79)
MODEL_NAME = "ANFIS + GA"


def _fit_restarts(
    listings: pd.DataFrame, device: torch.device, population: int, generations: int
) -> tuple[ANFIS, Pipeline, pd.DataFrame, int]:
    days = pd.Index(listings["listed_at"].unique()).sort_values()
    if len(days) < 2:
        raise ValueError("Genetic search requires at least two dates inside the training period.")
    validation_start = days[max(1, int(0.8 * len(days)))]
    before_validation = listings["listed_at"] < validation_start
    inner_train = listings.loc[before_validation]
    inner_validation = listings.loc[~before_validation]

    preprocessor = fit_preprocessor(inner_train)
    train = listing_tensors(inner_train, preprocessor, device)
    validation = listing_tensors(inner_validation, preprocessor, device)
    model = ANFIS(len(FEATURES)).to(device)

    solutions = {}
    for seed in SEEDS:
        solutions[seed] = search_memberships(model, train, validation, population, generations, seed)
    best_seed = min(SEEDS, key=lambda seed: solutions[seed][1][-1])
    chromosome, _ = solutions[best_seed]

    # Refit the rule outputs on the full outer training period. Keep the inner
    # scaler because the learned membership parameters use its coordinates.
    with torch.no_grad():
        model.centers.copy_(chromosome[0])
        model.log_widths.copy_(chromosome[1])
    model.fit_consequents(*listing_tensors(listings, preprocessor, device))

    history = pd.DataFrame(
        {
            "seed": seed,
            "generation": generation,
            "inner_validation_mae_rub": mae * PRICE_SCALE,
            "inner_validation_start": validation_start.date(),
            "selected_seed": seed == best_seed,
        }
        for seed, (_, scores) in solutions.items()
        for generation, mae in enumerate(scores, 1)
    )
    return model.eval(), preprocessor, history, best_seed


@torch.inference_mode()
def _predict(model: ANFIS, listings: pd.DataFrame, preprocessor: Pipeline) -> pd.Series:
    features, _ = listing_tensors(listings, preprocessor, model.centers.device)
    values = (model(features) * PRICE_SCALE).cpu().numpy()
    return pd.Series(values, index=listings.index, name=MODEL_NAME)


def run_genetic_experiment(
    source: Path,
    output_dir: Path,
    population: int,
    generations: int,
    folds: int,
    test_start: date,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the gpu extra and check the NVIDIA driver.")
    output_names = (
        "search.csv", "fold_metrics.csv", "test_metrics.csv", "model.pt", "validation.png", "test.png"
    )
    if source.resolve() in {(output_dir / name).resolve() for name in output_names}:
        raise ValueError("The source and output files must have different paths.")

    device = torch.device("cuda:0")
    print(f"Training on {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})", flush=True)
    listings = pd.read_csv(source, usecols=["listed_at", TARGET, *FEATURES])
    development, test = split_holdout(listings, test_start)
    fold_results = []
    search_history = []

    torch.cuda.synchronize(device)
    training_started = perf_counter()
    for fold, (train, validation) in enumerate(temporal_folds(development, folds), 1):
        fold_started = perf_counter()
        model, preprocessor, history, seed = _fit_restarts(train, device, population, generations)
        predicted = _predict(model, validation, preprocessor)
        validation_metrics = regression_metrics(validation[TARGET], predicted)
        train_mae = (train[TARGET] - _predict(model, train, preprocessor)).abs().mean()
        fold_results.append(
            {
                "fold": fold,
                "seed": seed,
                "period_start": validation["listed_at"].min().date(),
                "period_end": validation["listed_at"].max().date(),
                "train_listings": len(train),
                "validation_listings": len(validation),
                "train_mae_rub": train_mae,
                **validation_metrics,
                "training_seconds": perf_counter() - fold_started,
            }
        )
        search_history.append(history.assign(fold=fold))
        print(f"Fold {fold}/{folds}: validation MAE {validation_metrics['mae_rub']:,.0f} RUB", flush=True)

    model, preprocessor, history, seed = _fit_restarts(development, device, population, generations)
    torch.cuda.synchronize(device)
    training_seconds = perf_counter() - training_started
    search_history.append(history.assign(fold=pd.NA))
    predicted = _predict(model, test, preprocessor)
    metrics = pd.DataFrame(fold_results)
    search = pd.concat(search_history, ignore_index=True)
    test_metrics = pd.DataFrame([{
        "model": MODEL_NAME,
        "split": "test",
        "seed": seed,
        "validation_mae_rub": metrics["mae_rub"].mean(),
        "training_seconds": training_seconds,
        "listings": len(test),
        **regression_metrics(test[TARGET], predicted),
    }])

    output_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(output_dir / "search.csv", index=False)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    save_checkpoint(model, preprocessor, output_dir / "model.pt")
    plot_genetic_validation(search, metrics, output_dir / "validation.png")
    plot_test_predictions(test[[TARGET]].join(predicted), test_metrics, output_dir / "test.png")
    print(test_metrics.round(2).to_string(index=False))
