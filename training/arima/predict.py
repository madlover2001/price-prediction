from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.arima.config import MODEL_NAME
from training.common.registry import artifact_dir, get_product, read_json


def predict_product(product_id: str, province: str, steps: int) -> pd.DataFrame:
    product = get_product(product_id)
    directory = artifact_dir(product_id, MODEL_NAME)
    metadata = read_json(directory / "training_metadata.json")

    model_file = None
    for province_name, path in metadata["model_files"].items():
        if province_name.lower() == province.lower():
            model_file = ROOT_DIR / path
            province = province_name
            break
    if not model_file:
        raise ValueError(f"No hay modelo ARIMA para provincia: {province}")

    model = joblib.load(model_file)
    forecast = model.forecast(steps=steps)
    return pd.DataFrame(
        {
            "fecha": forecast.index,
            "producto": product.display_name,
            "provincia": province,
            "prediction": forecast.values,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predice pasos futuros con ARIMA/SARIMAX.")
    parser.add_argument("--product", required=True)
    parser.add_argument("--province", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_product(args.product, args.province, args.steps)
    if args.output:
        predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
