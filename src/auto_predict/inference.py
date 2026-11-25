"""Load the trained ANFIS and estimate a price from terminal input."""

from datetime import date
from math import expm1, isfinite, log1p
from pathlib import Path

import torch

from auto_predict.evaluation import FEATURES, GENERATION_FEATURE, TRANSMISSION_FEATURE
from auto_predict.models.anfis import ANFIS, MIN_MEMBERSHIP_WIDTH


def load_model(path: Path) -> tuple[ANFIS, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    features = checkpoint["features"]
    expected = (
        FEATURES,
        [*FEATURES, TRANSMISSION_FEATURE],
        [*FEATURES, TRANSMISSION_FEATURE, GENERATION_FEATURE],
    )
    if features not in expected:
        raise ValueError(f"Unexpected features in {path}.")
    terms = checkpoint["state_dict"]["centers"].shape[1]
    model = ANFIS(len(features), terms).eval()
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


def _read_model_year() -> int:
    current_year = date.today().year
    while True:
        year = _read_number("Год выпуска автомобиля (например, 2019): ", 1886)
        if year.is_integer() and year <= current_year:
            return int(year)
        print(f"Введите целый год выпуска до {current_year} включительно.")


def _read_automatic() -> int:
    while True:
        answer = input("Коробка передач (1 — АКПП, 0 — другая): ").strip()
        if answer in ("0", "1"):
            return int(answer)
        print("Введите 1 или 0.")


def _read_generation() -> float:
    while True:
        answer = input("Поколение Kia Rio (1–4, Enter — неизвестно): ").strip()
        if not answer:
            return float("nan")
        if answer in ("1", "2", "3", "4"):
            return float(answer)
        print("Введите число от 1 до 4 или нажмите Enter.")


def run_interactive(model_path: Path) -> int:
    try:
        model, checkpoint = load_model(model_path)
    except FileNotFoundError:
        print(f"Файл модели не найден: {model_path}")
        print("Сначала обучите ANFIS командой:")
        print(
            "uv run --extra gpu python scripts/train_anfis_genetic.py "
            "data/processed/listings.csv --lookback-months 2"
        )
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Ошибка загрузки модели: {error}")
        return 1

    print("Оценка цены объявления Kia Rio по модели ANFIS + GA.")
    encoding = checkpoint.get("feature_encoding", {})
    try:
        if "reference_year" in encoding:
            age = encoding["reference_year"] - _read_model_year()
        elif encoding.get("age_basis") == "continuous":
            model_year = _read_model_year()
            today = date.today()
            year_start, next_year = date(today.year, 1, 1), date(today.year + 1, 1, 1)
            age = today.year - model_year + (today - year_start) / (next_year - year_start)
        else:
            age = _read_number("Возраст автомобиля в годах (например, 6): ", 0)
        mileage = _read_number("Пробег в километрах (например, 97500): ", 0)
        power = _read_number("Мощность двигателя в лошадиных силах (например, 123): ", 1)
        has_transmission = TRANSMISSION_FEATURE in checkpoint["features"]
        has_generation = GENERATION_FEATURE in checkpoint["features"]
        vehicle_details = [_read_automatic()] if has_transmission else []
        if has_generation:
            vehicle_details.append(_read_generation())
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        return 1

    if encoding.get("log_mileage"):
        mileage = log1p(mileage / 10_000)
    raw = torch.tensor([[age, mileage, power, *vehicle_details]], dtype=torch.float64)
    features = standardize_features(raw, checkpoint)
    with torch.inference_mode():
        prediction = model(features)
        if checkpoint.get("log_target", False):
            prediction = prediction.exp()
        price = prediction.item() * checkpoint["price_scale"]
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
        rule_centers = centers[model.feature_indices, terms].tolist()
        age_center, mileage_center, power_center = rule_centers[:3]
        if encoding.get("log_mileage"):
            mileage_center = 10_000 * expm1(mileage_center)
        age_label = f"{age_center:.1f} лет"
        if "reference_year" in encoding:
            age_label = f"{encoding['reference_year'] - age_center:.1f} г. выпуска"
        gearbox_label = f", центр АКПП {rule_centers[3]:.2f}" if has_transmission else ""
        generation_label = f", центр поколения {rule_centers[4]:.1f}" if has_generation else ""
        print(
            f"  Правило {index + 1} ({weight:.1%}) — {age_label}, "
            f"{mileage_center:.0f} км, {power_center:.0f} л. с.{gearbox_label}{generation_label}"
        )
    return 0
