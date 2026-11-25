"""Evaluate genetic ANFIS with an inner search in each temporal split."""

from dataclasses import dataclass
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
    training_window,
)
from auto_predict.experiments.anfis import (
    PRICE_SCALE,
    fit_preprocessor,
    listing_tensors,
    save_checkpoint,
)
from auto_predict.models.anfis import ANFIS, CONSEQUENT_PENALTY
from auto_predict.models.anfis_genetic import search_memberships
from auto_predict.reports.plots import plot_genetic_validation, plot_test_predictions


SEEDS = (17, 42, 79)
MODEL_NAME = "ANFIS + GA"


@dataclass(frozen=True)
class GeneticFit:
    penalty: float = CONSEQUENT_PENALTY
    log_target: bool = False
    half_life_days: int = 0
    mutation_scale: float = 0.3
    terms: int = 2


def _recency_weights(
    listings: pd.DataFrame, half_life_days: int, device: torch.device
) -> torch.Tensor | None:
    if half_life_days == 0:
        return None
    age = (listings["listed_at"].max() - listings["listed_at"]).dt.days
    return torch.exp2(-torch.as_tensor(age.to_numpy(copy=True), device=device) / half_life_days)


def fit_restarts(
    listings: pd.DataFrame,
    device: torch.device,
    population: int,
    generations: int,
    options: GeneticFit = GeneticFit(),
    feature_names: list[str] | None = None,
) -> tuple[ANFIS, Pipeline, pd.DataFrame, int]:
    days = pd.Index(listings["listed_at"].unique()).sort_values()
    if len(days) < 2:
        raise ValueError("Genetic search requires at least two dates inside the training period.")
    validation_start = days[max(1, int(0.8 * len(days)))]
    before_validation = listings["listed_at"] < validation_start
    inner_train = listings.loc[before_validation]
    inner_validation = listings.loc[~before_validation]

    preprocessor = fit_preprocessor(inner_train, feature_names)
    train = listing_tensors(inner_train, preprocessor, device)
    validation = listing_tensors(inner_validation, preprocessor, device)
    model = ANFIS(train[0].shape[1], options.terms).to(device)
    weights = _recency_weights(inner_train, options.half_life_days, device)

    solutions = {}
    for seed in SEEDS:
        solutions[seed] = search_memberships(
            model, train, validation, population, generations, seed,
            penalty=options.penalty, log_target=options.log_target, weights=weights,
            mutation_scale=options.mutation_scale,
        )
    best_seed = min(SEEDS, key=lambda seed: solutions[seed][1][-1])
    chromosome, _ = solutions[best_seed]

    # Refit the rule outputs on the full outer training period. Keep the inner
    # scaler because the learned membership parameters use its coordinates.
    with torch.no_grad():
        model.centers.copy_(chromosome[0])
        model.log_widths.copy_(chromosome[1])
    features, target = listing_tensors(listings, preprocessor, device)
    if options.log_target:
        target = target.log()
    model.fit_consequents(
        features, target, penalty=options.penalty,
        weights=_recency_weights(listings, options.half_life_days, device),
    )

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
def predict_prices(
    model: ANFIS, listings: pd.DataFrame, preprocessor: Pipeline, *, log_target: bool = False
) -> pd.Series:
    features, _ = listing_tensors(listings, preprocessor, model.centers.device)
    prediction = model(features)
    if log_target:
        prediction = prediction.exp()
    values = (prediction * PRICE_SCALE).cpu().numpy()
    return pd.Series(values, index=listings.index, name=MODEL_NAME)


def _evaluate_window(
    development: pd.DataFrame,
    device: torch.device,
    population: int,
    generations: int,
    folds: int,
    months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_results = []
    search_history = []
    for fold, (train, validation) in enumerate(temporal_folds(development, folds), 1):
        fold_started = perf_counter()
        train = training_window(train, months)
        model, preprocessor, history, seed = fit_restarts(train, device, population, generations)
        predicted = predict_prices(model, validation, preprocessor)
        validation_metrics = regression_metrics(validation[TARGET], predicted)
        train_mae = (train[TARGET] - predict_prices(model, train, preprocessor)).abs().mean()
        fold_results.append(
            {
                "fold": fold,
                "seed": seed,
                "train_period_start": train["listed_at"].min().date(),
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
        print(
            f"Window {months or 'all'}, fold {fold}/{folds}: "
            f"validation MAE {validation_metrics['mae_rub']:,.0f} RUB",
            flush=True,
        )
    return pd.DataFrame(fold_results), pd.concat(search_history, ignore_index=True)


def run_genetic_experiment(
    source: Path,
    output_dir: Path,
    population: int,
    generations: int,
    folds: int,
    test_start: date,
    lookback_months: tuple[int, ...] = (0,),
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the gpu extra and check the NVIDIA driver.")
    if not lookback_months or min(lookback_months) < 0:
        raise ValueError("Training windows must be nonnegative. Use 0 for all history.")
    output_names = (
        "search.csv", "fold_metrics.csv", "test_metrics.csv", "model.pt",
        "validation.png", "test.png", "window_selection.csv",
    )
    if source.resolve() in {(output_dir / name).resolve() for name in output_names}:
        raise ValueError("The source and output files must have different paths.")

    device = torch.device("cuda:0")
    print(f"Training on {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})", flush=True)
    listings = pd.read_csv(source, usecols=["listed_at", TARGET, *FEATURES])
    development, test = split_holdout(listings, test_start)

    torch.cuda.synchronize(device)
    training_started = perf_counter()
    candidates = {
        months: _evaluate_window(development, device, population, generations, folds, months)
        for months in dict.fromkeys(lookback_months)
    }
    scores = pd.Series({
        months: metrics["mae_rub"].mean()
        for months, (metrics, _) in candidates.items()
    }, name="validation_mae_rub").rename_axis("lookback_months").sort_values(kind="stable")
    best_months = int(scores.index[0])
    metrics, search = candidates[best_months]
    print(f"Selected window: {best_months or 'all'} months by validation MAE", flush=True)

    final_train = training_window(development, best_months)
    model, preprocessor, history, seed = fit_restarts(final_train, device, population, generations)
    torch.cuda.synchronize(device)
    training_seconds = perf_counter() - training_started
    search = pd.concat([search, history.assign(fold=pd.NA)], ignore_index=True)
    predicted = predict_prices(model, test, preprocessor)
    test_metrics = pd.DataFrame([{
        "model": MODEL_NAME,
        "split": "test",
        "seed": seed,
        "lookback_months": best_months,
        "train_period_start": final_train["listed_at"].min().date(),
        "train_listings": len(final_train),
        "validation_mae_rub": metrics["mae_rub"].mean(),
        "training_seconds": training_seconds,
        "listings": len(test),
        **regression_metrics(test[TARGET], predicted),
    }])

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "window_selection.csv")
    search.to_csv(output_dir / "search.csv", index=False)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    save_checkpoint(model, preprocessor, output_dir / "model.pt")
    plot_genetic_validation(search, metrics, output_dir / "validation.png")
    plot_test_predictions(test[[TARGET]].join(predicted), test_metrics, output_dir / "test.png")
    print(test_metrics.round(2).to_string(index=False))
