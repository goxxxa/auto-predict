"""Train an ANFIS across temporal folds and evaluate the final model."""

from datetime import date
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from auto_predict.evaluation import (
    FEATURES,
    TARGET,
    regression_metrics,
    split_holdout,
    temporal_folds,
)
from auto_predict.models.anfis import ANFIS, train_anfis
from auto_predict.reports.plots import plot_learning_curves, plot_test_predictions


PRICE_SCALE = 1_000_000


def fit_preprocessor(listings: pd.DataFrame) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    ).fit(listings[FEATURES])


def listing_tensors(
    listings: pd.DataFrame, preprocessor: Pipeline, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    features = preprocessor.transform(listings[FEATURES])
    target = listings[TARGET].to_numpy() / PRICE_SCALE
    return (
        torch.as_tensor(features, dtype=torch.float32, device=device),
        torch.as_tensor(target, dtype=torch.float32, device=device),
    )


def save_checkpoint(model: ANFIS, preprocessor: Pipeline, output: Path) -> None:
    imputer = preprocessor.named_steps["simpleimputer"]
    scaler = preprocessor.named_steps["standardscaler"]
    torch.save(
        {
            "features": FEATURES,
            "price_scale": PRICE_SCALE,
            "feature_medians": torch.from_numpy(imputer.statistics_),
            "feature_means": torch.from_numpy(scaler.mean_),
            "feature_scales": torch.from_numpy(scaler.scale_),
            "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
        },
        output,
    )


def run_experiment(
    source: Path, output_dir: Path, epochs: int, folds: int, test_start: date
) -> None:
    if epochs < 1:
        raise ValueError("Training requires at least one epoch.")
    output_names = ("learning.csv", "test_metrics.csv", "validation.png", "test.png", "model.pt")
    if source.resolve() in {(output_dir / name).resolve() for name in output_names}:
        raise ValueError("The source and output files must have different paths.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the gpu extra and check the NVIDIA driver.")
    device = torch.device("cuda:0")
    print(f"Training on {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})", flush=True)

    listings = pd.read_csv(source, usecols=["listed_at", TARGET, *FEATURES])
    development, test = split_holdout(listings, test_start)
    fold_logs = []
    torch.cuda.synchronize(device)
    training_started = perf_counter()
    for fold, (train, validation) in enumerate(temporal_folds(development, folds), 1):
        preprocessor = fit_preprocessor(train)
        _, history = train_anfis(
            *listing_tensors(train, preprocessor, device),
            epochs=epochs,
            validation=listing_tensors(validation, preprocessor, device),
        )
        fold_logs.append(pd.DataFrame(history).assign(fold=fold))
        print(f"Fold {fold}/{folds}: {len(train):,} train, {len(validation):,} validation", flush=True)

    learning = pd.concat(fold_logs, ignore_index=True).rename(
        columns={"train_mae": "train_mae_rub", "validation_mae": "validation_mae_rub"}
    )
    learning[["train_mae_rub", "validation_mae_rub"]] *= PRICE_SCALE
    epoch_scores = learning.groupby("epoch")["validation_mae_rub"].mean()
    selected_epoch = int(epoch_scores.idxmin())
    print(f"Selected epoch {selected_epoch}: validation MAE {epoch_scores.min():,.0f} RUB", flush=True)

    preprocessor = fit_preprocessor(development)
    model, _ = train_anfis(
        *listing_tensors(development, preprocessor, device), epochs=selected_epoch
    )
    torch.cuda.synchronize(device)
    training_seconds = perf_counter() - training_started
    test_features, _ = listing_tensors(test, preprocessor, device)
    with torch.inference_mode():
        predicted = (model(test_features) * PRICE_SCALE).cpu().numpy()

    metrics = {
        "model": "ANFIS",
        "split": "test",
        "selected_epoch": selected_epoch,
        "validation_mae_rub": epoch_scores.loc[selected_epoch],
        "training_seconds": training_seconds,
        "listings": len(test),
        **regression_metrics(test[TARGET], predicted),
    }
    test_metrics = pd.DataFrame([metrics])
    test_predictions = test[[TARGET]].assign(ANFIS=predicted)

    output_dir.mkdir(parents=True, exist_ok=True)
    learning.to_csv(output_dir / "learning.csv", index=False)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    plot_learning_curves(learning, selected_epoch, output_dir / "validation.png")
    plot_test_predictions(test_predictions, test_metrics, output_dir / "test.png")

    save_checkpoint(model, preprocessor, output_dir / "model.pt")
    print(f"Test MAE {metrics['mae_rub']:,.0f} RUB on {len(test):,} listings")
    print(f"Saved model, metrics and plots to {output_dir}")
