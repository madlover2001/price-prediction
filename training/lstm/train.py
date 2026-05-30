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
from training.common.registry import artifact_dir, selected_products, write_model_result
from training.lstm.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS,
    MODEL_NAME,
    RANDOM_SEED,
    TRAIN_RATIO,
    VERBOSE,
    WINDOW_SIZE,
)


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


def build_model(input_shape):
    tf = _import_tensorflow()
    tf.random.set_seed(RANDOM_SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    feature_scaler,
    target_scaler,
):
    X_train, y_train, X_test, y_test, prediction_rows = [], [], [], [], []

    for _, group in df.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        feature_values = feature_scaler.transform(group[feature_columns])
        target_values = target_scaler.transform(group[[target_column]]).reshape(-1)
        split_index = int(len(group) * TRAIN_RATIO)
        split_index = max(WINDOW_SIZE + 1, min(split_index, len(group) - 1))
        rows = group.reset_index(drop=True)

        for index in range(WINDOW_SIZE - 1, len(group)):
            sequence = feature_values[index - WINDOW_SIZE + 1:index + 1]
            target = target_values[index]
            if index < split_index:
                X_train.append(sequence)
                y_train.append(target)
            else:
                X_test.append(sequence)
                y_test.append(target)
                prediction_rows.append(rows.iloc[index])

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        pd.DataFrame(prediction_rows).reset_index(drop=True),
    )


def inverse_target(values, target_scaler) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def train_product(product_id: str) -> dict:
    _import_tensorflow()
    from sklearn.preprocessing import MinMaxScaler
    from training.common.metrics import regression_metrics

    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product)

    train_rows = []
    for _, group in bundle.data.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        split_index = int(len(group) * TRAIN_RATIO)
        split_index = max(WINDOW_SIZE + 1, min(split_index, len(group) - 1))
        train_rows.append(group.iloc[:split_index])
    train_fit_df = pd.concat(train_rows, ignore_index=True)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_fit_df[bundle.feature_columns])
    target_scaler.fit(train_fit_df[[bundle.target_column]])

    X_train, y_train, X_test, y_test_scaled, prediction_rows = build_sequences(
        bundle.data,
        bundle.feature_columns,
        bundle.target_column,
        feature_scaler,
        target_scaler,
    )
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_true = inverse_target(y_test_scaled, target_scaler)
    y_pred = inverse_target(y_pred_scaled, target_scaler)
    metrics = regression_metrics(y_true, y_pred)

    predictions = prediction_rows[["fecha", "producto", "provincia"]].copy()
    predictions["y_true"] = y_true
    predictions["y_pred"] = y_pred
    predictions["model_name"] = MODEL_NAME
    predictions["product_id"] = product.product_id

    directory = artifact_dir(product.product_id, MODEL_NAME)
    model_path = directory / "model.keras"
    feature_scaler_path = directory / "feature_scaler.joblib"
    target_scaler_path = directory / "target_scaler.joblib"
    model.save(model_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    metadata = {
        "product_id": product.product_id,
        "product_name": product.display_name,
        "model_name": MODEL_NAME,
        "model_file": str(model_path.relative_to(ROOT_DIR)),
        "feature_scaler_file": str(feature_scaler_path.relative_to(ROOT_DIR)),
        "target_scaler_file": str(target_scaler_path.relative_to(ROOT_DIR)),
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "window_size": WINDOW_SIZE,
        "train_ratio": TRAIN_RATIO,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": LSTM_UNITS,
        "learning_rate": LEARNING_RATE,
        "verbose": VERBOSE,
    }
    write_model_result(product.product_id, MODEL_NAME, metrics, metadata, predictions)
    return {"product_id": product.product_id, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena LSTM por producto.")
    parser.add_argument("--product", default="all", help="Producto a entrenar o 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for product in selected_products(args.product):
        results.append(train_product(product.product_id))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
