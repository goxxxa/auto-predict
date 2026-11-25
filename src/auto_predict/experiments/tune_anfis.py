"""Select ANFIS hyperparameters using only the pre-test temporal folds."""

from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from auto_predict.evaluation import (
    FEATURES, GENERATION_FEATURE, TARGET, TRANSMISSION_FEATURE,
    regression_metrics, split_holdout, temporal_folds,
    training_window, training_window_days,
)
from auto_predict.experiments.anfis import save_checkpoint
from auto_predict.experiments.anfis_genetic import GeneticFit, fit_restarts, predict_prices
from auto_predict.reports.plots import plot_test_predictions


@dataclass(frozen=True)
class Trial:
    # Zero reproduces the existing two-calendar-month checkpoint.
    days: int = 0
    population: int = 20
    generations: int = 35
    fit: GeneticFit = field(default_factory=GeneticFit)
    age_basis: str = "calendar"
    log_mileage: bool = False
    transmission: bool = False
    generation: bool = False


def encode_features(listings: pd.DataFrame, trial: Trial, reference_year: int) -> pd.DataFrame:
    age = listings["age_years"]
    mileage = listings["mileage_km"]
    if trial.age_basis == "cohort":
        age = reference_year - listings["model_year"]
    elif trial.age_basis == "continuous":
        dates = listings["listed_at"].dt
        age = age + (dates.day_of_year - 1) / (365 + dates.is_leap_year.astype(int))
    if trial.log_mileage:
        mileage = np.log1p(mileage / 10_000)
    encoded = listings.assign(age_years=age, mileage_km=mileage)
    if trial.transmission:
        transmission = listings["transmission"]
        encoded[TRANSMISSION_FEATURE] = transmission.eq("АКПП").astype(float).where(
            transmission.notna()
        )
    return encoded


def _trial_rows(listings: pd.DataFrame, days: int) -> pd.DataFrame:
    if days == 0:
        return training_window(listings, 2)
    return training_window_days(listings, days)


def fit_trial(train: pd.DataFrame, trial: Trial, device: torch.device):
    features = [*FEATURES, TRANSMISSION_FEATURE] if trial.transmission else FEATURES
    if trial.generation:
        features = [*features, GENERATION_FEATURE]
    return fit_restarts(
        _trial_rows(train, trial.days), device, trial.population, trial.generations,
        trial.fit, feature_names=features,
    )


def trial_from_row(row: pd.Series) -> Trial:
    return Trial(
        days=int(row["days"]),
        population=int(row["population"]),
        generations=int(row["generations"]),
        fit=GeneticFit(
            penalty=float(row["penalty"]), log_target=bool(row["log_target"]),
            half_life_days=int(row["half_life_days"]), mutation_scale=float(row["mutation_scale"]),
            terms=int(row["terms"]),
        ),
        age_basis=row["age_basis"], log_mileage=bool(row["log_mileage"]),
        transmission=bool(row.get("transmission", False)),
        generation=bool(row.get("generation", False)),
    )


def run_tuning(
    source: Path, output: Path, test_start: date, folds: int,
    refine_from: Path | None = None, add_generation: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ANFIS tuning.")
    if add_generation and refine_from is None:
        raise ValueError("Use --refine-from when testing the generation feature.")
    device = torch.device("cuda:0")
    listings = pd.read_csv(source)
    development, test = split_holdout(listings, test_start)
    split_cache = {}
    results = {}

    def evaluate(trial: Trial, stage: str) -> None:
        if trial in results:
            return
        encoding = (trial.age_basis, trial.log_mileage, trial.transmission)
        if encoding not in split_cache:
            encoded = encode_features(development, trial, test_start.year)
            split_cache[encoding] = list(temporal_folds(encoded, folds))
        errors = []
        for train, validation in split_cache[encoding]:
            model, scaler, _, _ = fit_trial(train, trial, device)
            predicted = predict_prices(model, validation, scaler, log_target=trial.fit.log_target)
            errors.append((validation[TARGET] - predicted).abs().mean())
        config = asdict(trial)
        settings = config.pop("fit")
        results[trial] = {
            "stage": stage, **config, **settings,
            "validation_mae_rub": sum(errors) / len(errors),
            **{f"fold_{index}_mae_rub": value for index, value in enumerate(errors, 1)},
        }
        if len(results) % 5 == 0:
            leader = min(row["validation_mae_rub"] for row in results.values())
            print(f"ANFIS trial {len(results)} ({stage}): best MAE {leader:,.0f} RUB", flush=True)

    def best_trials(stage: str, count: int) -> list[Trial]:
        eligible = [trial for trial, row in results.items() if row["stage"] == stage]
        return sorted(eligible, key=lambda trial: results[trial]["validation_mae_rub"])[:count]

    if refine_from is None:
        evaluate(Trial(), "reference")
        for days in (14, 21, 30, 45, 60, 75, 90, 120):
            for log_target in (False, True):
                trial = Trial(days=days, population=16, generations=20)
                evaluate(replace(trial, fit=replace(trial.fit, log_target=log_target)), "window_target")

        for trial in best_trials("window_target", 4):
            for penalty in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03):
                evaluate(
                    replace(trial, population=20, generations=35,
                            fit=replace(trial.fit, penalty=penalty)),
                    "regularization",
                )

        for trial in best_trials("regularization", 2):
            for half_life in (7, 14, 30, 60):
                for days in dict.fromkeys((trial.days, max(trial.days, 90))):
                    evaluate(
                        replace(trial, days=days, fit=replace(trial.fit, half_life_days=half_life)),
                        "time_decay",
                    )

        leaders = best_trials("regularization", 2) + best_trials("time_decay", 1)
        for trial in leaders:
            for mutation in (0.15, 0.3, 0.55):
                evaluate(
                    replace(trial, population=32, generations=70,
                            fit=replace(trial.fit, mutation_scale=mutation)),
                    "search_budget",
                )

        for trial in best_trials("regularization", 1) + best_trials("time_decay", 1):
            for age_basis in ("calendar", "continuous", "cohort"):
                for log_mileage in (False, True):
                    evaluate(replace(trial, age_basis=age_basis, log_mileage=log_mileage), "features")

        for trial in best_trials("features", 2):
            for half_life in (3, 5, 10, 14):
                evaluate(replace(trial, fit=replace(trial.fit, half_life_days=half_life)), "decay_refinement")
            for penalty in (0.0001, 0.0003, 0.001):
                evaluate(replace(trial, fit=replace(trial.fit, terms=3, penalty=penalty)), "rule_count")
    else:
        if "transmission" not in listings:
            raise ValueError("Refinement needs the extended CSV with the transmission column.")
        if add_generation and GENERATION_FEATURE not in listings:
            raise ValueError("Refinement needs the extended CSV with the generation column.")
        reference = pd.read_csv(refine_from).nsmallest(1, "validation_mae_rub").iloc[0]
        leader = trial_from_row(reference)
        evaluate(leader, "reference")
        automatic = replace(leader, transmission=True, generation=add_generation)
        for days in (30, 60, 90):
            for half_life in (5, 10, 14, 21):
                evaluate(
                    replace(automatic, days=days,
                            fit=replace(automatic.fit, half_life_days=half_life)),
                    "transmission_window",
                )
        leader = best_trials("transmission_window", 1)[0]
        for penalty in (0.00003, 0.0003, 0.001):
            evaluate(replace(leader, fit=replace(leader.fit, penalty=penalty)),
                     "transmission_penalty")
        if add_generation:
            evaluate(replace(leader, fit=replace(leader.fit, log_target=True)), "generation_log_target")

    best = min(results, key=lambda trial: results[trial]["validation_mae_rub"])
    selected = results[best]
    print(f"Chosen ANFIS: {selected}", flush=True)
    development = encode_features(development, best, test_start.year)
    test = encode_features(test, best, test_start.year)
    model, scaler, _, seed = fit_trial(development, best, device)
    prediction = predict_prices(model, test, scaler, log_target=best.fit.log_target)
    relative_error = (prediction - test[TARGET]).abs() / test[TARGET]
    summary = pd.DataFrame([{
        "model": "Tuned ANFIS + GA", **selected, "seed": seed,
        "train_listings": len(_trial_rows(development, best.days)),
        "test_listings": len(test),
        **regression_metrics(test[TARGET], prediction),
        "mape_percent": 100 * relative_error.mean(),
        "within_20_percent": 100 * relative_error.le(0.2).mean(),
        "split": "test",
    }])

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results.values()).sort_values("validation_mae_rub").to_csv(
        output / "validation_trials.csv", index=False
    )
    summary.to_csv(output / "test_metrics.csv", index=False)
    feature_encoding = {"age_basis": best.age_basis, "log_mileage": best.log_mileage}
    if best.age_basis == "cohort":
        feature_encoding["reference_year"] = test_start.year
    save_checkpoint(
        model, scaler, output / "model.pt", log_target=best.fit.log_target,
        feature_encoding=feature_encoding,
    )
    plot_test_predictions(
        test[[TARGET]].join(prediction.rename("Tuned ANFIS + GA")), summary, output / "test.png"
    )
    print(summary.round(2).to_string(index=False))
