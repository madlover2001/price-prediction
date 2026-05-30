from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.data_loader import load_product_dataset
from training.common.registry import artifact_dir, get_product, read_json
from training.lstm.config import MODEL_NAME


def _import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(
            "TensorFlow no pudo cargarse correctamente. Instala o repara las dependencias con "
            "`pip install -r training/lstm/requirements.txt`. En Windows tambien puede requerir "
            "un runtime compatible de Microsoft Visual C++."
        ) from exc
    return tf


def predict_product(product_id: str, province: str | None = None) -> pd.DataFrame:
    tf = _import_tensorflow()
    product = get_product(product_id)
    bundle = load_product_dataset(product)
    directory = artifact_dir(product_id, MODEL_NAME)
    metadata = read_json(directory / "training_metadata.json")
    model = tf.keras.models.load_model(directory / "model.keras")
    feature_scaler = joblib.load(directory / "feature_scaler.joblib")
    target_scaler = joblib.load(directory / "target_scaler.joblib")

    df = bundle.data.copy()
    if province:
        df = df.loc[df["provincia"].str.lower() == province.lower()].copy()

    feature_columns = metadata["feature_columns"]
    window_size = metadata["window_size"]
    rows = []

    for _, group in df.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        values = feature_scaler.transform(group[feature_columns])
        group_rows = group.reset_index(drop=True)
        for index in range(window_size - 1, len(group)):
            X = values[index - window_size + 1:index + 1].reshape(1, window_size, len(feature_columns))
            pred_scaled = model.predict(X, verbose=0).reshape(-1)[0]
            prediction = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            rows.append(
                {
                    "fecha": group_rows.loc[index, "fecha"],
                    "producto": group_rows.loc[index, "producto"],
                    "provincia": group_rows.loc[index, "provincia"],
                    "prediction": prediction,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predice con un modelo LSTM entrenado.")
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
