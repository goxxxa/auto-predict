"""Compare the trained models and explain the Sugeno rules in original units."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from auto_predict.anfis import ANFIS
from auto_predict.evaluation import FEATURES, TARGET, split_holdout
from auto_predict.inference import load_model, membership_parameters, standardize_features
from auto_predict.reporting import plot_model_comparison


def _plot_memberships(
    variants: dict, listings: pd.DataFrame, output: Path
) -> None:
    figure = Figure(figsize=(11, 6), layout="constrained")
    axes = figure.subplots(2, 3, sharex="col", sharey=True)
    labels = ("Возраст, лет", "Пробег, тыс. км", "Мощность, л. с.")
    units = (1, 1_000, 1)
    for row, (name, (model, checkpoint)) in enumerate(variants.items()):
        model_centers, model_widths = membership_parameters(model, checkpoint)
        centers, widths = model_centers.numpy(), model_widths.numpy()
        for feature, axis in enumerate(axes[row]):
            low, high = listings[FEATURES[feature]].quantile([0.01, 0.99])
            domain = np.linspace(low, high, 200)
            for term, label in zip(centers[feature].argsort(), ("Нижний центр", "Верхний центр")):
                offset = (domain - centers[feature, term]) / widths[feature, term]
                axis.plot(domain / units[feature], np.exp(-0.5 * offset**2), label=label)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(labels[feature])
        axes[row, 0].set_ylabel(name)
    axes[0, 0].legend()
    figure.suptitle("Функции принадлежности в диапазоне 1–99-го процентилей теста")
    figure.savefig(output, dpi=160)


@torch.inference_mode()
def _explain_rules(
    model: ANFIS, checkpoint: dict, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = torch.from_numpy(test[FEATURES].to_numpy(dtype=float, copy=True))
    filled = torch.where(raw.isnan(), checkpoint["feature_medians"], raw)
    means = checkpoint["feature_means"]
    scales = checkpoint["feature_scales"]
    features = standardize_features(raw, checkpoint)

    weights = model.rule_weights(features)
    price_scale = checkpoint["price_scale"]
    terms = model.rule_terms.numpy()
    centers, _ = membership_parameters(model, checkpoint)
    lower_terms = centers.numpy().argmin(axis=1)
    table = pd.DataFrame(
        np.where(terms == lower_terms, "lower center", "upper center"), columns=FEATURES
    ).rename(columns={feature: f"{feature}_term" for feature in FEATURES})
    table.insert(0, "rule", np.arange(1, len(terms) + 1))
    table["mean_weight"] = weights.mean(dim=0).numpy()
    table["strongest_share"] = (
        weights.argmax(dim=1).bincount(minlength=len(terms)) / len(weights)
    ).numpy()

    coefficients = model.consequents.reshape(len(terms), -1) * price_scale
    slopes = coefficients[:, 1:] / scales
    table["intercept_rub"] = (coefficients[:, 0] - slopes @ means).numpy()
    for column, feature in enumerate(FEATURES):
        table[f"rub_per_{feature}"] = slopes[:, column].numpy()

    examples = []
    target = test[TARGET].to_numpy()
    for quantile in (0.1, 0.5, 0.9):
        row = np.abs(target - np.quantile(target, quantile)).argmin()
        x = features[row : row + 1]
        contributions = model.rule_contributions(x)[0].numpy() * price_scale
        example = table.iloc[:, : len(FEATURES) + 1].assign(
            price_quantile=quantile,
            actual_price_rub=target[row],
            predicted_price_rub=model(x).item() * price_scale,
            weight=weights[row].numpy(),
            weighted_contribution_rub=contributions,
        )
        for column, feature in enumerate(FEATURES):
            example[feature] = filled[row, column].item()
        examples.append(example.sort_values("weight", ascending=False))
    return table, pd.concat(examples, ignore_index=True)


def write_comparison(
    source: Path, runs_dir: Path, output_dir: Path, test_start: date
) -> None:
    listings = pd.read_csv(source, usecols=["listed_at", TARGET, *FEATURES])
    _, test = split_holdout(listings, test_start)
    baseline_dir = runs_dir / "anfis"
    genetic_dir = runs_dir / "anfis_genetic"
    classic = pd.read_csv(runs_dir / "baselines.csv")
    classic_folds = classic.loc[classic.split.eq("validation"), ["model", "fold", "mae_rub"]]
    test_results = pd.concat(
        [
            classic.loc[classic.split.eq("test")],
            pd.read_csv(baseline_dir / "test_metrics.csv"),
            pd.read_csv(genetic_dir / "test_metrics.csv"),
        ],
        ignore_index=True,
    )
    if not test_results["listings"].eq(len(test)).all():
        raise ValueError("The reported test sizes do not match the selected holdout.")

    epoch = test_results.loc[test_results.model.eq("ANFIS"), "selected_epoch"].item()
    learning = pd.read_csv(baseline_dir / "learning.csv")
    anfis_folds = learning.loc[learning.epoch.eq(epoch), ["fold", "validation_mae_rub"]]
    anfis_folds = anfis_folds.rename(columns={"validation_mae_rub": "mae_rub"}).assign(model="ANFIS")
    genetic_folds = pd.read_csv(genetic_dir / "fold_metrics.csv", usecols=["fold", "mae_rub"])
    genetic_folds = genetic_folds.assign(model="ANFIS + GA")
    folds = pd.concat([classic_folds, anfis_folds, genetic_folds], ignore_index=True)
    fold_counts = folds.groupby("model")["fold"].nunique()
    if fold_counts.nunique() != 1:
        raise ValueError("The runs have different numbers of validation folds.")

    metrics = test_results.set_index("model")[
        ["mae_rub", "rmse_rub", "median_ae_rub", "training_seconds"]
    ].sort_values("mae_rub")
    metrics = metrics.rename(columns={"mae_rub": "test_mae_rub", "rmse_rub": "test_rmse_rub",
                                      "median_ae_rub": "test_median_ae_rub"})
    metrics.insert(0, "validation_mae_rub", folds.groupby("model")["mae_rub"].mean())

    variants = {
        "ANFIS": load_model(baseline_dir / "model.pt"),
        "ANFIS + GA": load_model(genetic_dir / "model.pt"),
    }
    rules, examples = _explain_rules(*variants["ANFIS + GA"], test)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "comparison.csv")
    folds.to_csv(output_dir / "fold_comparison.csv", index=False)
    rules.to_csv(output_dir / "rules.csv", index=False)
    examples.to_csv(output_dir / "examples.csv", index=False)
    _plot_memberships(variants, test, output_dir / "memberships.png")
    plot_model_comparison(folds, metrics, output_dir / "comparison.png")
    print(metrics.round(2).to_string())
