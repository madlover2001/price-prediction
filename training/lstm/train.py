from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.config import FEATURE_SETS, HOLDOUT_TEST_RATIO
from training.common.data_loader import load_product_dataset
from training.common.registry import artifact_dir, selected_products, write_model_result
from training.common.splits import apply_holdout_cutoffs, compute_holdout_cutoffs, expanding_validation_windows
from training.lstm.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS,
    MODEL_NAME,
    RANDOM_SEED,
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
    cutoffs: dict,
):
    """Construye secuencias de longitud WINDOW_SIZE por provincia. `cutoffs` mapea
    provincia -> fecha de corte: filas con fecha < corte van a train, >= corte a test/val.
    Usar una sola funcion para el holdout externo y para las ventanas de validacion
    garantiza que las secuencias de la porcion "posterior" siempre puedan mirar hacia
    atras dentro de la porcion "anterior" para completar su ventana, igual que hacia el
    codigo original."""
    X_train, y_train, X_holdout, y_holdout, prediction_rows = [], [], [], [], []

    for province, group in df.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        cutoff = cutoffs.get(province)
        if cutoff is None or len(group) < WINDOW_SIZE:
            continue
        feature_values = feature_scaler.transform(group[feature_columns])
        target_values = target_scaler.transform(group[[target_column]]).reshape(-1)
        rows = group.reset_index(drop=True)
        dates = rows["fecha"]

        for index in range(WINDOW_SIZE - 1, len(group)):
            sequence = feature_values[index - WINDOW_SIZE + 1 : index + 1]
            target = target_values[index]
            if dates.iloc[index] < cutoff:
                X_train.append(sequence)
                y_train.append(target)
            else:
                X_holdout.append(sequence)
                y_holdout.append(target)
                prediction_rows.append(rows.iloc[index])

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(X_holdout, dtype=np.float32),
        np.asarray(y_holdout, dtype=np.float32),
        pd.DataFrame(prediction_rows).reset_index(drop=True),
    )


def inverse_target(values, target_scaler) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def _fit_scalers(train_only_df: pd.DataFrame, feature_columns: list[str], target_column: str):
    from sklearn.preprocessing import MinMaxScaler

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_only_df[feature_columns])
    target_scaler.fit(train_only_df[[target_column]])
    return feature_scaler, target_scaler


def _validation_windows_lstm(dev_df: pd.DataFrame, feature_columns: list[str], target_column: str) -> list[dict]:
    """Ventanas de validacion expansivas (ver training/common/splits.py). Corrige el punto
    #2 del docente: LSTM no tenia ningun conjunto de validacion independiente."""
    from training.common.metrics import regression_metrics

    results = []
    for train_window, val_window in expanding_validation_windows(dev_df):
        combined = pd.concat([train_window, val_window], ignore_index=True)
        val_start = val_window["fecha"].min()
        cutoffs = {province: val_start for province in combined["provincia"].unique()}

        fit_only = combined[combined["fecha"] < val_start]
        if fit_only.empty:
            continue
        feature_scaler, target_scaler = _fit_scalers(fit_only, feature_columns, target_column)

        X_train, y_train, X_val, y_val, _ = build_sequences(
            combined, feature_columns, target_column, feature_scaler, target_scaler, cutoffs
        )
        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
        y_pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
        y_true = inverse_target(y_val, target_scaler)
        y_pred = inverse_target(y_pred_scaled, target_scaler)
        results.append(regression_metrics(y_true, y_pred))

    return results


def train_product(product_id: str, feature_set: str) -> dict:
    _import_tensorflow()
    from training.common.metrics import regression_metrics

    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product, feature_set=feature_set)

    cutoffs = compute_holdout_cutoffs(bundle.data, test_ratio=HOLDOUT_TEST_RATIO)
    train_df, _ = apply_holdout_cutoffs(bundle.data, cutoffs)

    window_metrics = _validation_windows_lstm(train_df, bundle.feature_columns, bundle.target_column)
    validation_rmse_mean = (
        float(np.mean([m["rmse"] for m in window_metrics])) if window_metrics else math.nan
    )

    feature_scaler, target_scaler = _fit_scalers(train_df, bundle.feature_columns, bundle.target_column)
    X_train, y_train, X_test, y_test_scaled, prediction_rows = build_sequences(
        bundle.data, bundle.feature_columns, bundle.target_column, feature_scaler, target_scaler, cutoffs
    )
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_true = inverse_target(y_test_scaled, target_scaler)
    y_pred = inverse_target(y_pred_scaled, target_scaler)
    metrics = regression_metrics(y_true, y_pred)
    metrics["validation_rmse_mean"] = validation_rmse_mean

    predictions = prediction_rows[["fecha", "producto", "provincia"]].copy()
    predictions["y_true"] = y_true
    predictions["y_pred"] = y_pred
    predictions["model_name"] = MODEL_NAME
    predictions["product_id"] = product.product_id
    predictions["feature_set"] = feature_set

    directory = artifact_dir(product.product_id, MODEL_NAME, feature_set)
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
        "feature_set": feature_set,
        "model_file": str(model_path.relative_to(ROOT_DIR)),
        "feature_scaler_file": str(feature_scaler_path.relative_to(ROOT_DIR)),
        "target_scaler_file": str(target_scaler_path.relative_to(ROOT_DIR)),
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "window_size": WINDOW_SIZE,
        "holdout_test_ratio": HOLDOUT_TEST_RATIO,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": LSTM_UNITS,
        "learning_rate": LEARNING_RATE,
        "verbose": VERBOSE,
        "validation_rmse_mean": validation_rmse_mean,
        "validation_windows_results": window_metrics,
        "n_validation_windows": len(window_metrics),
    }
    write_model_result(product.product_id, MODEL_NAME, feature_set, metrics, metadata, predictions)
    return {"product_id": product.product_id, "feature_set": feature_set, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena LSTM por producto.")
    parser.add_argument("--product", default="all", help="Producto a entrenar o 'all'.")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for product in selected_products(args.product):
        for feature_set in args.feature_sets:
            results.append(train_product(product.product_id, feature_set))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
