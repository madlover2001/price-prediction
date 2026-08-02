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
from training.common.data_loader import load_product_dataset
from training.common.horizon import build_future_exog_frame
from training.common.registry import artifact_dir, get_product, read_json


def predict_product(product_id: str, province: str, steps: int, feature_set: str = "full") -> pd.DataFrame:
    product = get_product(product_id)
    directory = artifact_dir(product_id, MODEL_NAME, feature_set)
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
    exog_columns = metadata.get("exog_columns", [])

    future_exog = None
    if exog_columns:
        # feature_set="full": las exogenas futuras se desconocen, se propagan al ultimo
        # valor observado (ver docs/correcciones_docente.md, punto 5).
        bundle = load_product_dataset(product, feature_set=feature_set)
        province_df = bundle.data.loc[bundle.data["provincia"] == province].sort_values("fecha")
        last_row = province_df.iloc[-1]
        last_date = last_row["fecha"]
        future_exog = build_future_exog_frame(last_row, last_date, steps, exog_columns)

    forecast = model.forecast(steps=steps, exog=future_exog)
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
    parser.add_argument("--feature-set", default="full", choices=["base", "full"])
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_product(args.product, args.province, args.steps, args.feature_set)
    if args.output:
        predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
