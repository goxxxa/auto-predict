"""Load the trained ANFIS and estimate a price from terminal input."""

from math import isfinite
from pathlib import Path

import torch

from auto_predict.anfis import ANFIS, MIN_MEMBERSHIP_WIDTH
from auto_predict.evaluation import FEATURES


def load_model(path: Path) -> tuple[ANFIS, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES:
        raise ValueError(f"Unexpected features in {path}.")
    model = ANFIS(len(FEATURES)).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def standardize_features(raw: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    filled = torch.where(raw.isnan(), checkpoint["feature_medians"], raw)
    return ((filled - checkpoint["feature_means"]) / checkpoint["feature_scales"]).float()


def membership_parameters(
    model: ANFIS, checkpoint: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    means = checkpoint["feature_means"][:, None]
    scales = checkpoint["feature_scales"][:, None]
    centers = model.centers.detach() * scales + means
    widths = (model.log_widths.detach().exp() + MIN_MEMBERSHIP_WIDTH) * scales
    return centers, widths


def _read_number(prompt: str, minimum: int) -> float:
    while True:
        value = input(prompt).replace(" ", "").replace(",", ".")
        try:
            number = float(value)
        except ValueError:
            print("Введите число.")
            continue
        if isfinite(number) and number >= minimum:
            return number
        print(f"Введите конечное число не меньше {minimum}.")


def run_interactive(model_path: Path) -> int:
    try:
        model, checkpoint = load_model(model_path)
    except FileNotFoundError:
        print(f"Файл модели не найден: {model_path}")
        print("Сначала обучите ANFIS командой:")
        print(
            "uv run --extra gpu python scripts/train_anfis_genetic.py "
            "data/processed/listings.csv"
        )
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Ошибка загрузки модели: {error}")
        return 1

    print("Оценка цены объявления Kia Rio по модели ANFIS + GA.")
    try:
        age = _read_number("Возраст автомобиля в годах (например, 6): ", 0)
        mileage = _read_number("Пробег в километрах (например, 97500): ", 0)
        power = _read_number("Мощность двигателя в лошадиных силах (например, 123): ", 1)
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        return 1

    raw = torch.tensor([[age, mileage, power]], dtype=torch.float64)
    features = standardize_features(raw, checkpoint)
    with torch.inference_mode():
        price = model(features).item() * checkpoint["price_scale"]
        weights, indices = model.rule_weights(features)[0].topk(3)
    if not isfinite(price) or price <= 0:
        print("Для введённых характеристик модель не смогла рассчитать положительную цену.")
        return 1

    price_label = f"{price:,.0f}".replace(",", " ")
    print(f"\nПрогноз цены объявления: {price_label} руб.")
    print("Три правила с наибольшим весом и центры их функций принадлежности.")
    centers, _ = membership_parameters(model, checkpoint)
    for weight, index in zip(weights.tolist(), indices.tolist()):
        terms = model.rule_terms[index]
        age_center, mileage_center, power_center = centers[model.feature_indices, terms].tolist()
        print(
            f"  Правило {index + 1} ({weight:.1%}) — {age_center:.1f} лет, "
            f"{mileage_center:.0f} км, {power_center:.0f} л. с."
        )
    return 0
