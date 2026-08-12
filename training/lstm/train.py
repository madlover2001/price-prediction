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
from training.common.horizon import recursive_forecast
from training.common.registry import artifact_dir, selected_products, write_model_result
from training.common.splits import apply_holdout_cutoffs, compute_holdout_cutoffs, expanding_validation_windows
from training.lstm.config import (
    BATCH_SIZE,
    EPOCHS,
    HYPERPARAM_GRID,
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


def build_model(input_shape, lstm_units: int = LSTM_UNITS, learning_rate: float = LEARNING_RATE):
    tf = _import_tensorflow()
    tf.random.set_seed(RANDOM_SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    feature_scaler,
    target_scaler,
    cutoffs: dict,
    window_size: int = WINDOW_SIZE,
):
    """Construye secuencias de longitud `window_size` por provincia. `cutoffs` mapea
    provincia -> fecha de corte: filas con fecha < corte van a train, >= corte a test/val.
    Usar una sola funcion para el holdout externo y para las ventanas de validacion
    garantiza que las secuencias de la porcion "posterior" siempre puedan mirar hacia
    atras dentro de la porcion "anterior" para completar su ventana, igual que hacia el
    codigo original."""
    X_train, y_train, X_holdout, y_holdout, prediction_rows = [], [], [], [], []

    for province, group in df.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        cutoff = cutoffs.get(province)
        if cutoff is None or len(group) < window_size:
            continue
        feature_values = feature_scaler.transform(group[feature_columns])
        target_values = target_scaler.transform(group[[target_column]]).reshape(-1)
        rows = group.reset_index(drop=True)
        dates = rows["fecha"]

        for index in range(window_size - 1, len(group)):
            sequence = feature_values[index - window_size + 1 : index + 1]
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


def _window_metrics_for_hyperparams(
    dev_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    lstm_units: int,
) -> list[dict]:
    """Ventanas de validacion expansivas (ver training/common/splits.py). Corrige el punto
    #2 del docente: LSTM no tenia ningun conjunto de validacion independiente. Se usa aqui
    tambien para C.8: elegir hiperparametros con un proceso reproducible."""
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
            combined, feature_columns, target_column, feature_scaler, target_scaler, cutoffs, window_size
        )
        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = build_model((X_train.shape[1], X_train.shape[2]), lstm_units, LEARNING_RATE)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
        y_pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
        y_true = inverse_target(y_val, target_scaler)
        y_pred = inverse_target(y_pred_scaled, target_scaler)
        results.append(regression_metrics(y_true, y_pred))

    return results


def select_best_hyperparams(
    dev_df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[dict, float, list[dict]]:
    tuning_results = []
    for combo in HYPERPARAM_GRID:
        window_metrics = _window_metrics_for_hyperparams(
            dev_df, feature_columns, target_column, combo["window_size"], combo["lstm_units"]
        )
        mean_rmse = float(np.mean([m["rmse"] for m in window_metrics])) if window_metrics else math.nan
        tuning_results.append({**combo, "validation_rmse_mean": mean_rmse, "n_windows": len(window_metrics)})

    valid_results = [row for row in tuning_results if not math.isnan(row["validation_rmse_mean"])]
    if not valid_results:
        return {"window_size": WINDOW_SIZE, "lstm_units": LSTM_UNITS}, math.nan, tuning_results

    best = min(valid_results, key=lambda row: row["validation_rmse_mean"])
    return {"window_size": best["window_size"], "lstm_units": best["lstm_units"]}, best["validation_rmse_mean"], tuning_results


def _oof_and_operational(
    dev_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    lstm_units: int,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Con los hiperparametros ya elegidos: genera predicciones OOF por ventana de
    validacion (insumo para la ventana comun de validacion entre familias, punto C.3) y,
    sobre la ULTIMA ventana, un pronostico operacional h=1..3 por provincia, recursivo,
    con exogenas al ultimo valor conocido (misma logica de training/common/horizon.py) --
    insumo para el criterio de seleccion de modelo, punto C.6/C.7."""
    windows = expanding_validation_windows(dev_df)
    oof_rows = []
    operational_errors: list[tuple] = []

    for window_index, (train_window, val_window) in enumerate(windows):
        combined = pd.concat([train_window, val_window], ignore_index=True)
        val_start = val_window["fecha"].min()
        cutoffs = {province: val_start for province in combined["provincia"].unique()}

        fit_only = combined[combined["fecha"] < val_start]
        if fit_only.empty:
            continue
        feature_scaler, target_scaler = _fit_scalers(fit_only, feature_columns, target_column)

        X_train, y_train, X_val, y_val, prediction_rows = build_sequences(
            combined, feature_columns, target_column, feature_scaler, target_scaler, cutoffs, window_size
        )
        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = build_model((X_train.shape[1], X_train.shape[2]), lstm_units, LEARNING_RATE)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
        y_pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
        y_true = inverse_target(y_val, target_scaler)
        y_pred = inverse_target(y_pred_scaled, target_scaler)

        for i in range(len(prediction_rows)):
            row = prediction_rows.iloc[i]
            oof_rows.append(
                {
                    "fecha": row["fecha"],
                    "producto": row["producto"],
                    "provincia": row["provincia"],
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                }
            )

        if window_index == len(windows) - 1:

            def predict_one_step(feature_row: dict, working_history: pd.DataFrame) -> float:
                context = working_history.tail(window_size - 1)
                window_df = pd.concat([context, pd.DataFrame([feature_row])], ignore_index=True)
                if len(window_df) < window_size:
                    return float("nan")
                values = feature_scaler.transform(window_df[feature_columns])
                X = values.reshape(1, window_size, len(feature_columns))
                pred_scaled = model.predict(X, verbose=0).reshape(-1)[0]
                return float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])

            for provincia, province_val_df in val_window.groupby("provincia", sort=False):
                history = fit_only[fit_only["provincia"] == provincia].sort_values("fecha")
                if len(history) < window_size:
                    continue
                forecast_df = recursive_forecast(predict_one_step, history, feature_columns, 3, target_column)
                actual_by_date = province_val_df.set_index("fecha")[target_column]
                for _, row in forecast_df.iterrows():
                    target_date = row["fecha"]
                    if target_date in actual_by_date.index and not math.isnan(row["y_pred"]):
                        operational_errors.append(
                            (int(row["horizonte"]), float(actual_by_date.loc[target_date]), float(row["y_pred"]))
                        )

    return pd.DataFrame(oof_rows), operational_errors


def train_product(product_id: str, feature_set: str) -> dict:
    _import_tensorflow()
    from training.common.metrics import regression_metrics

    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product, feature_set=feature_set)

    cutoffs = compute_holdout_cutoffs(bundle.data, test_ratio=HOLDOUT_TEST_RATIO)
    train_df, _ = apply_holdout_cutoffs(bundle.data, cutoffs)

    best_hyperparams, validation_rmse_mean, tuning_results = select_best_hyperparams(
        train_df, bundle.feature_columns, bundle.target_column
    )
    window_size = best_hyperparams["window_size"]
    lstm_units = best_hyperparams["lstm_units"]

    oof_predictions, operational_errors = _oof_and_operational(
        train_df, bundle.feature_columns, bundle.target_column, window_size, lstm_units
    )
    if operational_errors:
        y_true_op = np.array([row[1] for row in operational_errors], dtype=float)
        y_pred_op = np.array([row[2] for row in operational_errors], dtype=float)
        operational_rmse_mean = float(np.sqrt(np.mean((y_true_op - y_pred_op) ** 2)))
    else:
        operational_rmse_mean = math.nan

    feature_scaler, target_scaler = _fit_scalers(train_df, bundle.feature_columns, bundle.target_column)
    X_train, y_train, X_test, y_test_scaled, prediction_rows = build_sequences(
        bundle.data, bundle.feature_columns, bundle.target_column, feature_scaler, target_scaler, cutoffs, window_size
    )
    model = build_model((X_train.shape[1], X_train.shape[2]), lstm_units, LEARNING_RATE)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_true = inverse_target(y_test_scaled, target_scaler)
    y_pred = inverse_target(y_pred_scaled, target_scaler)
    metrics = regression_metrics(y_true, y_pred)
    metrics["validation_rmse_mean"] = validation_rmse_mean
    metrics["operational_rmse_mean"] = operational_rmse_mean

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

    if not oof_predictions.empty:
        oof_predictions["model_name"] = MODEL_NAME
        oof_predictions["product_id"] = product.product_id
        oof_predictions["feature_set"] = feature_set
    validation_predictions_path = directory / "validation_predictions.csv"
    oof_predictions.to_csv(validation_predictions_path, index=False, encoding="utf-8-sig")

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
        "window_size": window_size,
        "holdout_test_ratio": HOLDOUT_TEST_RATIO,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": lstm_units,
        "learning_rate": LEARNING_RATE,
        "verbose": VERBOSE,
        "hyperparam_grid": HYPERPARAM_GRID,
        "hyperparam_tuning_results": tuning_results,
        "validation_rmse_mean": validation_rmse_mean,
        "operational_rmse_mean": operational_rmse_mean,
        "validation_predictions_file": str(validation_predictions_path.relative_to(ROOT_DIR)),
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
