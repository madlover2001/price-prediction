from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.arima.config import (
    MIN_OBSERVATIONS,
    MODEL_NAME,
    ORDER,
    ORDER_GRID,
    PRODUCT_ORDER_OVERRIDES,
    SEASONAL_ORDER,
    SEASONAL_ORDER_GRID,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from training.common.data_loader import load_product_dataset
from training.common.metrics import regression_metrics
from training.common.registry import artifact_dir, selected_products, write_json, write_model_result


def _import_sarimax():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError(
            "statsmodels no esta instalado. Instala dependencias con "
            "`pip install -r training/arima/requirements.txt`."
        ) from exc
    return SARIMAX


def fit_sarimax(SARIMAX, series: pd.Series, order: tuple, seasonal_order: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def select_best_order(SARIMAX, train_series: pd.Series) -> tuple[tuple, tuple, list[dict]]:
    split_index = int(len(train_series) * VALIDATION_RATIO)
    split_index = max(1, min(split_index, len(train_series) - 1))
    fit_series = train_series.iloc[:split_index]
    validation_series = train_series.iloc[split_index:]
    tuning_results = []

    for order in ORDER_GRID:
        for seasonal_order in SEASONAL_ORDER_GRID:
            try:
                fitted = fit_sarimax(SARIMAX, fit_series, order, seasonal_order)
                forecast = fitted.forecast(steps=len(validation_series))
                metrics = regression_metrics(validation_series.values, forecast.values)
                score = metrics["rmse"]
            except Exception:
                score = math.inf
                metrics = {"rmse": math.inf, "mae": math.inf}

            tuning_results.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "validation_rmse": score,
                    "validation_mae": metrics["mae"],
                }
            )

    valid_results = [row for row in tuning_results if np.isfinite(row["validation_rmse"])]
    if not valid_results:
        return ORDER, SEASONAL_ORDER, tuning_results

    best = min(valid_results, key=lambda row: (row["validation_rmse"], row["validation_mae"]))
    return best["order"], best["seasonal_order"], tuning_results


def train_product(product_id: str) -> dict:
    SARIMAX = _import_sarimax()
    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product)
    directory = artifact_dir(product.product_id, MODEL_NAME)
    province_metrics = []
    prediction_parts = []
    model_files = {}
    selected_orders = {}
    product_override = PRODUCT_ORDER_OVERRIDES.get(product.product_id)

    for province, group in bundle.data.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        if len(group) < MIN_OBSERVATIONS:
            continue

        series = group.set_index("fecha")[bundle.target_column].asfreq("MS").interpolate().ffill().bfill()
        split_index = int(len(series) * TRAIN_RATIO)
        split_index = max(1, min(split_index, len(series) - 1))
        train_series = series.iloc[:split_index]
        test_series = series.iloc[split_index:]
        if product_override:
            best_order = product_override["order"]
            best_seasonal_order = product_override["seasonal_order"]
            tuning_results = []
            selection_strategy = "product_order_override"
        else:
            best_order, best_seasonal_order, tuning_results = select_best_order(SARIMAX, train_series)
            selection_strategy = "validation_grid_search"

        fitted = fit_sarimax(SARIMAX, train_series, best_order, best_seasonal_order)
        forecast = fitted.forecast(steps=len(test_series))

        current_metrics = regression_metrics(test_series.values, forecast.values)
        current_metrics["provincia"] = province
        province_metrics.append(current_metrics)

        final_fitted = fit_sarimax(SARIMAX, series, best_order, best_seasonal_order)

        province_key = str(province).lower().replace(" ", "_")
        model_path = directory / f"model_{province_key}.joblib"
        joblib.dump(final_fitted, model_path)
        model_files[str(province)] = str(model_path.relative_to(ROOT_DIR))
        selected_orders[str(province)] = {
            "order": best_order,
            "seasonal_order": best_seasonal_order,
            "tuning_results": tuning_results,
        }

        predictions = pd.DataFrame(
            {
                "fecha": test_series.index,
                "producto": product.display_name,
                "provincia": province,
                "y_true": test_series.values,
                "y_pred": forecast.values,
                "model_name": MODEL_NAME,
                "product_id": product.product_id,
            }
        )
        prediction_parts.append(predictions)

    if not prediction_parts:
        raise ValueError(f"No hubo provincias con suficientes observaciones para {product.product_id}")

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = regression_metrics(all_predictions["y_true"], all_predictions["y_pred"])
    province_metrics_path = directory / "province_metrics.csv"
    pd.DataFrame(province_metrics).to_csv(province_metrics_path, index=False, encoding="utf-8-sig")

    metadata = {
        "product_id": product.product_id,
        "product_name": product.display_name,
        "model_name": MODEL_NAME,
        "model_files": model_files,
        "target_column": bundle.target_column,
        "train_ratio": TRAIN_RATIO,
        "order": ORDER,
        "seasonal_order": SEASONAL_ORDER,
        "order_grid": ORDER_GRID,
        "seasonal_order_grid": SEASONAL_ORDER_GRID,
        "validation_ratio_within_train": VALIDATION_RATIO,
        "selection_strategy": selection_strategy,
        "selected_orders_by_province": selected_orders,
        "min_observations": MIN_OBSERVATIONS,
        "province_metrics_file": str(province_metrics_path.relative_to(ROOT_DIR)),
    }
    write_json(directory / "config.json", metadata)
    write_model_result(product.product_id, MODEL_NAME, metrics, metadata, all_predictions)
    return {"product_id": product.product_id, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena ARIMA/SARIMAX por producto y provincia.")
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
