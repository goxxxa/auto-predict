"""Оценка цены Kia Rio. Введите возраст в годах, пробег в км и мощность в л. с."""

from pathlib import Path

from auto_predict.inference import run_interactive


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "data/processed/anfis_genetic/model.pt"
    raise SystemExit(run_interactive(model_path))
