from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.config import FEATURE_SETS, HOLDOUT_TEST_RATIO
from training.common.data_loader import load_product_dataset
from training.common.horizon import recursive_forecast
from training.common.metrics import regression_metrics
from training.common.registry import artifact_dir, selected_products, write_model_result
from training.common.splits import apply_holdout_cutoffs, compute_holdout_cutoffs, expanding_validation_windows
from training.xgboost.config import (
    ENABLE_PARAM_TUNING,
    LAG_BLEND_FEATURE,
    LAG_BLEND_WEIGHT_GRID,
    MODEL_NAME,
    PARAM_GRID,
    PARAMS,
)


def blend_with_lag(raw_prediction, df: pd.DataFrame, blend_weight: float):
    if not blend_weight:
        return raw_prediction
    return (1 - blend_weight) * raw_prediction + blend_weight * df[LAG_BLEND_FEATURE].to_numpy()


def _validation_rmse(
    params: dict, blend_weight: float, windows: list, feature_columns: list[str], target_column: str
) -> tuple[float, float]:
    window_rmses, window_maes = [], []
    for fit_df, validation_df in windows:
        model = XGBRegressor(**params)
        model.fit(fit_df[feature_columns], fit_df[target_column])
        raw_pred = model.predict(validation_df[feature_columns])
        blended_pred = blend_with_lag(raw_pred, validation_df, blend_weight)
        window_metrics = regression_metrics(validation_df[target_column], blended_pred)
        window_rmses.append(window_metrics["rmse"])
        window_maes.append(window_metrics["mae"])
    if not window_rmses:
        return math.nan, math.nan
    return float(np.mean(window_rmses)), float(np.mean(window_maes))


def select_best_params(
    dev_df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[dict, float, float, list[dict]]:
    windows = expanding_validation_windows(dev_df)
    tuning_results = []

    # Busqueda conjunta de (hiperparametros, peso de blend): el modelo validado debe ser
    # exactamente el modelo reportado/desplegado (punto C.2). Si blend_weight=0.0 gana,
    # el blend queda desactivado sin necesidad de una opcion aparte.
    for index, params in enumerate(PARAM_GRID):
        for blend_weight in LAG_BLEND_WEIGHT_GRID:
            mean_rmse, mean_mae = _validation_rmse(params, blend_weight, windows, feature_columns, target_column)
            tuning_results.append(
                {
                    "candidate": index,
                    "blend_weight": blend_weight,
                    "validation_rmse_mean": mean_rmse,
                    "validation_mae_mean": mean_mae,
                    "n_windows": len(windows),
                    "params": params,
                }
            )

    valid_results = [row for row in tuning_results if not math.isnan(row["validation_rmse_mean"])]
    if not valid_results:
        return PARAMS, LAG_BLEND_WEIGHT_GRID[0], math.nan, tuning_results

    best = min(valid_results, key=lambda row: (row["validation_rmse_mean"], row["validation_mae_mean"]))
    return best["params"], best["blend_weight"], best["validation_rmse_mean"], tuning_results


def _oof_and_operational(
    params: dict,
    blend_weight: float,
    windows: list,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Con (hiperparametros, blend) ya elegidos: genera predicciones OOF (una por cada
    fila de cada ventana de validacion; las ventanas son disjuntas en fecha, asi que la
    concatenacion cubre el 80% de desarrollo sin solapes -- insumo para la ventana comun
    de validacion entre familias, punto C.3) y, sobre la ULTIMA ventana, un pronostico
    operacional h=1..3 por provincia, recursivo, con exogenas al ultimo valor conocido
    (misma logica de training/common/horizon.py) -- insumo para el criterio de seleccion
    de modelo, punto C.6/C.7."""
    oof_rows = []
    operational_errors: list[tuple] = []

    for window_index, (fit_df, validation_df) in enumerate(windows):
        model = XGBRegressor(**params)
        model.fit(fit_df[feature_columns], fit_df[target_column])
        raw_pred = model.predict(validation_df[feature_columns])
        blended_pred = blend_with_lag(raw_pred, validation_df, blend_weight)

        for fecha, producto, provincia, y_true, y_pred in zip(
            validation_df["fecha"],
            validation_df["producto"],
            validation_df["provincia"],
            validation_df[target_column],
            blended_pred,
        ):
            oof_rows.append(
                {
                    "fecha": fecha,
                    "producto": producto,
                    "provincia": provincia,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }
            )

        if window_index == len(windows) - 1:
            def predict_one_step(feature_row: dict, _working_history: pd.DataFrame) -> float:
                frame = pd.DataFrame([feature_row])[feature_columns]
                raw = float(model.predict(frame)[0])
                if blend_weight and LAG_BLEND_FEATURE in feature_row:
                    return (1 - blend_weight) * raw + blend_weight * feature_row[LAG_BLEND_FEATURE]
                return raw

            for provincia, province_val_df in validation_df.groupby("provincia", sort=False):
                history = fit_df[fit_df["provincia"] == provincia].sort_values("fecha")
                if history.empty:
                    continue
                forecast_df = recursive_forecast(
                    predict_one_step, history, feature_columns, 3, target_column
                )
                actual_by_date = province_val_df.set_index("fecha")[target_column]
                for _, row in forecast_df.iterrows():
                    target_date = row["fecha"]
                    if target_date in actual_by_date.index:
                        operational_errors.append(
                            (int(row["horizonte"]), float(actual_by_date.loc[target_date]), float(row["y_pred"]))
                        )

    return pd.DataFrame(oof_rows), operational_errors


def train_product(product_id: str, feature_set: str) -> dict:
    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product, feature_set=feature_set)

    cutoffs = compute_holdout_cutoffs(bundle.data, test_ratio=HOLDOUT_TEST_RATIO)
    train_df, test_df = apply_holdout_cutoffs(bundle.data, cutoffs)
    windows = expanding_validation_windows(train_df)

    if ENABLE_PARAM_TUNING:
        best_params, best_blend_weight, validation_rmse_mean, tuning_results = select_best_params(
            train_df, bundle.feature_columns, bundle.target_column
        )
    else:
        best_blend_weight = LAG_BLEND_WEIGHT_GRID[0]
        validation_rmse_mean, _ = _validation_rmse(
            PARAMS, best_blend_weight, windows, bundle.feature_columns, bundle.target_column
        )
        best_params, tuning_results = PARAMS, []

    oof_predictions, operational_errors = _oof_and_operational(
        best_params, best_blend_weight, windows, bundle.feature_columns, bundle.target_column
    )
    if operational_errors:
        y_true_op = np.array([row[1] for row in operational_errors], dtype=float)
        y_pred_op = np.array([row[2] for row in operational_errors], dtype=float)
        operational_rmse_mean = float(np.sqrt(np.mean((y_true_op - y_pred_op) ** 2)))
    else:
        operational_rmse_mean = math.nan

    model = XGBRegressor(**best_params)
    model.fit(train_df[bundle.feature_columns], train_df[bundle.target_column])

    raw_pred = model.predict(test_df[bundle.feature_columns])
    y_pred = blend_with_lag(raw_pred, test_df, best_blend_weight)
    metrics = regression_metrics(test_df[bundle.target_column], y_pred)
    metrics["validation_rmse_mean"] = validation_rmse_mean
    metrics["operational_rmse_mean"] = operational_rmse_mean

    predictions = test_df[["fecha", "producto", "provincia", bundle.target_column]].copy()
    predictions = predictions.rename(columns={bundle.target_column: "y_true"})
    predictions["y_pred_raw"] = raw_pred
    predictions["y_pred"] = y_pred
    predictions["model_name"] = MODEL_NAME
    predictions["product_id"] = product.product_id
    predictions["feature_set"] = feature_set

    directory = artifact_dir(product.product_id, MODEL_NAME, feature_set)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)

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
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "holdout_test_ratio": HOLDOUT_TEST_RATIO,
        "params": best_params,
        "tuning_results": tuning_results,
        "validation_rmse_mean": validation_rmse_mean,
        "operational_rmse_mean": operational_rmse_mean,
        "n_validation_windows": len(windows),
        "enable_param_tuning": ENABLE_PARAM_TUNING,
        "lag_blend_feature": LAG_BLEND_FEATURE,
        "lag_blend_weight": best_blend_weight,
        "lag_blend_weight_grid": LAG_BLEND_WEIGHT_GRID,
        "validation_predictions_file": str(validation_predictions_path.relative_to(ROOT_DIR)),
    }
    write_model_result(product.product_id, MODEL_NAME, feature_set, metrics, metadata, predictions)
    return {"product_id": product.product_id, "feature_set": feature_set, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena XGBoost por producto.")
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
