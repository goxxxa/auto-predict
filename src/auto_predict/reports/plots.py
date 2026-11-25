"""Plots of model training, validation errors, and test predictions."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from auto_predict.evaluation import TARGET


def plot_fold_errors(metrics: pd.DataFrame, output: Path) -> None:
    """Plot training and validation MAE for each chronological fold."""
    folds = metrics.loc[metrics["fold"].notna()]
    figure = Figure(figsize=(10, 7), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    for axis, (model, rows) in zip(axes.flat, folds.groupby("model", sort=False)):
        for split, label in (("train", "Обучение"), ("validation", "Валидация")):
            values = rows.loc[rows["split"] == split]
            axis.plot(
                values["fold"], values["mae_rub"] / 1_000, marker="o", label=label
            )
        axis.set_title(model)
        axis.set_xticks(rows["fold"].unique().to_numpy(dtype=int))
        axis.grid(alpha=0.25)

    axes[0, 0].legend()
    figure.suptitle("Ошибка на временных фолдах")
    figure.supxlabel("Номер фолда")
    figure.supylabel("MAE, тыс. ₽")
    figure.savefig(output, dpi=160)


def plot_test_predictions(
    predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    """Plot predicted vs. observed prices on a common scale for all baselines."""
    prices = predictions / 1_000_000
    actual = prices[TARGET]
    lower = min(0, prices.min().min())
    upper = prices.max().max() * 1.05
    errors = metrics.loc[metrics["split"] == "test"].set_index("model")["mae_rub"]
    model_names = prices.columns.drop(TARGET)
    columns = min(2, len(model_names))
    rows = (len(model_names) + columns - 1) // columns
    figure = Figure(figsize=(5 * columns + 1, 4 * rows + 1), layout="constrained")
    axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    for axis, name in zip(axes.flat, model_names):
        axis.scatter(actual, prices[name], s=5, alpha=0.15, color="#2364aa")
        axis.plot([lower, upper], [lower, upper], "--", color="#5a5a5a", linewidth=1)
        axis.set(xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
        axis.set_title(f"{name} · MAE {errors[name] / 1_000:.0f} тыс. ₽")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(model_names):]:
        axis.set_visible(False)

    figure.suptitle("Цены объявлений и прогнозы на отложенном тесте")
    figure.supxlabel("Цена объявления, млн ₽")
    figure.supylabel("Прогноз, млн ₽")
    figure.savefig(output, dpi=160)


def plot_learning_curves(
    history: pd.DataFrame, selected_epoch: int, output: Path
) -> None:
    """Show train and validation MAE by epoch for each fold and their mean."""
    columns = ["train_mae_rub", "validation_mae_rub"]
    curves = [
        (f"Фолд {fold}", rows.set_index("epoch")[columns])
        for fold, rows in history.groupby("fold")
    ]
    curves.append(("Среднее по фолдам", history.groupby("epoch")[columns].mean()))

    rows = (len(curves) + 2) // 3
    figure = Figure(figsize=(12, 3.2 * rows), layout="constrained")
    axes = figure.subplots(rows, 3, sharex=True, sharey=True, squeeze=False)
    for axis, (title, curve) in zip(axes.flat, curves):
        axis.plot(curve.index, curve[columns[0]] / 1_000, label="Обучение")
        axis.plot(curve.index, curve[columns[1]] / 1_000, label="Валидация")
        axis.axvline(selected_epoch, linestyle="--", linewidth=1, color="gray")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(curves):]:
        axis.set_visible(False)
    axes[0, 0].legend()
    figure.suptitle(f"Обучение ANFIS · выбрана эпоха {selected_epoch}")
    figure.supxlabel("Эпоха")
    figure.supylabel("MAE, тыс. ₽")
    figure.savefig(output, dpi=160)


def plot_genetic_validation(
    history: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    search_axis, fold_axis = figure.subplots(1, 2)
    outer_runs = history.loc[history["fold"].notna()]
    average_runs = outer_runs.groupby(["seed", "generation"])["inner_validation_mae_rub"].mean()
    for seed, curve in average_runs.groupby(level="seed"):
        search_axis.plot(
            curve.index.get_level_values("generation"), curve / 1_000, label=f"Seed {seed}"
        )
    search_axis.set(title="Внутренняя валидация", xlabel="Поколение", ylabel="MAE, тыс. ₽")
    fold_axis.plot(metrics["fold"], metrics["train_mae_rub"] / 1_000, "o-", label="Обучение")
    fold_axis.plot(metrics["fold"], metrics["mae_rub"] / 1_000, "o-", label="Валидация")
    fold_axis.set(title="Внешняя валидация", xlabel="Временной фолд", ylabel="MAE, тыс. ₽")
    fold_axis.set_xticks(metrics["fold"])
    for axis in (search_axis, fold_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=160)


def plot_model_comparison(folds: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    figure = Figure(figsize=(12, 5), layout="constrained")
    validation_axis, test_axis = figure.subplots(1, 2)
    for model, rows in folds.groupby("model"):
        validation_axis.plot(rows["fold"], rows["mae_rub"] / 1_000, "o-", label=model)
    validation_axis.set(title="Валидация", xlabel="Временной фолд", ylabel="MAE, тыс. ₽")
    validation_axis.set_xticks(sorted(folds["fold"].unique()))
    validation_axis.grid(alpha=0.25)
    validation_axis.legend()

    bars = test_axis.barh(metrics.index, metrics["test_mae_rub"] / 1_000, color="#2364aa")
    test_axis.bar_label(bars, fmt="%.0f", padding=3)
    test_axis.set_xlim(0, metrics["test_mae_rub"].max() / 1_000 * 1.18)
    test_axis.invert_yaxis()
    test_axis.set(title="Отложенный тест", xlabel="MAE, тыс. ₽")
    figure.savefig(output, dpi=160)
