"""Оценка цены Kia Rio по характеристикам автомобиля."""

from pathlib import Path

from auto_predict.inference import run_interactive


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "data/processed"
    runs = (
        "tuning/anfis_vehicle", "tuning/anfis_extended", "tuning/anfis",
        "anfis_genetic_recent", "anfis_genetic",
    )
    models = [output_dir / run / "model.pt" for run in runs]
    model_path = next((path for path in models if path.is_file()), models[-1])
    raise SystemExit(run_interactive(model_path))
