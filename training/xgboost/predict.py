from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.data_loader import load_product_dataset
from training.common.registry import artifact_dir, get_product, read_json
from training.xgboost.config import MODEL_NAME


def predict_product(product_id: str, province: str | None = None) -> pd.DataFrame:
    product = get_product(product_id)
    bundle = load_product_dataset(product)
    metadata = read_json(artifact_dir(product_id, MODEL_NAME) / "training_metadata.json")
    model = joblib.load(artifact_dir(product_id, MODEL_NAME) / "model.joblib")

    df = bundle.data.copy()
    if province:
        df = df.loc[df["provincia"].str.lower() == province.lower()].copy()
    raw_prediction = model.predict(df[metadata["feature_columns"]])
    blend_weight = metadata.get("lag_blend_weight", 0)
    blend_feature = metadata.get("lag_blend_feature")
    if blend_feature and blend_feature in df.columns:
        df["prediction"] = (1 - blend_weight) * raw_prediction + blend_weight * df[blend_feature].to_numpy()
    else:
        df["prediction"] = raw_prediction
    return df[["fecha", "producto", "provincia", "prediction"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predice con un modelo XGBoost entrenado.")
    parser.add_argument("--product", required=True)
    parser.add_argument("--province")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_product(args.product, args.province)
    if args.output:
        predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    else:
        print(predictions.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
