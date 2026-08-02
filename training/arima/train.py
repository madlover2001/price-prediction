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
    SEASONAL_ORDER,
    SEASONAL_ORDER_GRID,
)
from training.common.config import FEATURE_SETS, HOLDOUT_TEST_RATIO, MIN_TEST_OBS_FOR_PROVINCE_CONCLUSIONS
from training.common.data_loader import load_product_dataset
from training.common.horizon import CALENDAR_FEATURES, TRUE_EXOG_BASE_COLUMNS
from training.common.metrics import regression_metrics
from training.common.registry import artifact_dir, selected_products, write_json, write_model_result
from training.common.splits import compute_holdout_cutoffs, expanding_validation_windows

# Variables exogenas "verdaderas" que se le pasan a SARIMAX en feature_set="full" (no se
# incluyen los rezagos del propio target: el orden AR/estacional ya captura esa memoria).
EXOG_COLUMNS = TRUE_EXOG_BASE_COLUMNS + CALENDAR_FEATURES


def _import_sarimax():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError(
            "statsmodels no esta instalado. Instala dependencias con "
            "`pip install -r training/arima/requirements.txt`."
        ) from exc
    return SARIMAX


def fit_sarimax(SARIMAX, series: pd.Series, order: tuple, seasonal_order: tuple, exog: pd.DataFrame | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def _prepare_series(df: pd.DataFrame, target_column: str, exog_columns: list[str]):
    ordered = df.sort_values("fecha").set_index("fecha").asfreq("MS")
    series = ordered[target_column].interpolate().ffill().bfill()
    exog = None
    if exog_columns:
        exog = ordered[exog_columns].interpolate().ffill().bfill()
    return series, exog


def _is_stable_forecast(forecast_values, reference_values, max_multiple: float = 25.0) -> bool:
    """Con enforce_stationarity=False, algunos ordenes del grid producen pronosticos que
    divergen numericamente (raices AR/estacionales fuera del circulo unitario). Se
    descartan aqui en vez de dejarlos contaminar la seleccion de orden o las metricas
    finales con RMSE de escala irreal (ver docs/correcciones_docente.md, nota tecnica)."""
    forecast_values = np.asarray(forecast_values, dtype=float)
    if forecast_values.size == 0 or not np.all(np.isfinite(forecast_values)):
        return False
    reference_values = np.asarray(reference_values, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    if reference_values.size == 0:
        return True
    reference_scale = max(float(np.max(np.abs(reference_values))), 1e-6)
    return bool(np.all(np.abs(forecast_values) <= max_multiple * reference_scale))


def select_best_order(
    SARIMAX,
    train_df: pd.DataFrame,
    target_column: str,
    exog_columns: list[str],
) -> tuple[tuple, tuple, float, list[dict], list[dict]]:
    windows = expanding_validation_windows(train_df)
    tuning_results = []

    for order in ORDER_GRID:
        for seasonal_order in SEASONAL_ORDER_GRID:
            window_rmses = []
            for fit_window, val_window in windows:
                try:
                    fit_series, fit_exog = _prepare_series(fit_window, target_column, exog_columns)
                    val_series, val_exog = _prepare_series(val_window, target_column, exog_columns)
                    fitted = fit_sarimax(SARIMAX, fit_series, order, seasonal_order, exog=fit_exog)
                    forecast = fitted.forecast(steps=len(val_series), exog=val_exog)
                    if not _is_stable_forecast(forecast.values, fit_series.values):
                        window_rmses.append(math.inf)
                        continue
                    window_rmses.append(regression_metrics(val_series.values, forecast.values)["rmse"])
                except Exception:
                    window_rmses.append(math.inf)
            mean_rmse = float(np.mean(window_rmses)) if window_rmses else math.inf
            tuning_results.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "validation_rmse_mean": mean_rmse,
                    "n_windows": len(windows),
                }
            )

    valid_results = sorted(
        (row for row in tuning_results if np.isfinite(row["validation_rmse_mean"])),
        key=lambda row: row["validation_rmse_mean"],
    )
    if not valid_results:
        return ORDER, SEASONAL_ORDER, math.nan, tuning_results, []

    best = valid_results[0]
    return best["order"], best["seasonal_order"], best["validation_rmse_mean"], tuning_results, valid_results


def train_product(product_id: str, feature_set: str) -> dict:
    SARIMAX = _import_sarimax()
    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product, feature_set=feature_set)
    directory = artifact_dir(product.product_id, MODEL_NAME, feature_set)

    exog_columns = EXOG_COLUMNS if feature_set == "full" else []
    cutoffs = compute_holdout_cutoffs(bundle.data, test_ratio=HOLDOUT_TEST_RATIO)
    # "base" y "full" seleccionan su orden SIEMPRE mediante la misma busqueda en grid
    # validada por ventanas expansivas (nunca via los overrides manuales que se usaban
    # antes solo para "base"): usar un procedimiento fijo para un brazo de la ablacion y
    # uno validado para el otro habria sesgado la comparacion del punto #1 del docente.
    selection_strategy = "validation_grid_search"

    province_metrics = []
    prediction_parts = []
    model_files = {}
    selected_orders = {}
    validation_rmse_by_province = []

    for province, group in bundle.data.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        if len(group) < MIN_OBSERVATIONS:
            continue
        cutoff = cutoffs.get(province)
        if cutoff is None:
            continue

        train_part = group[group["fecha"] < cutoff]
        test_part = group[group["fecha"] >= cutoff]
        if train_part.empty or test_part.empty:
            continue

        best_order, best_seasonal_order, validation_rmse, tuning_results, ranked_candidates = select_best_order(
            SARIMAX, train_part, bundle.target_column, exog_columns
        )

        train_series, train_exog = _prepare_series(train_part, bundle.target_column, exog_columns)
        test_series, test_exog = _prepare_series(test_part, bundle.target_column, exog_columns)

        # El orden ganador en validacion puede seguir siendo inestable sobre el horizonte
        # de test (mas largo que las ventanas de validacion). Se prueban los siguientes
        # candidatos rankeados hasta encontrar un pronostico numericamente estable; si
        # ninguno lo es, se cae al orden conservador por defecto (sin componente
        # estacional) como ultimo recurso.
        candidates = ranked_candidates or [{"order": ORDER, "seasonal_order": SEASONAL_ORDER}]
        forecast = None
        fallback_used = False
        for rank, candidate in enumerate(candidates):
            order = tuple(candidate["order"])
            seasonal_order = tuple(candidate["seasonal_order"])
            try:
                candidate_fitted = fit_sarimax(SARIMAX, train_series, order, seasonal_order, exog=train_exog)
                candidate_forecast = candidate_fitted.forecast(steps=len(test_series), exog=test_exog)
            except Exception:
                continue
            if _is_stable_forecast(candidate_forecast.values, train_series.values):
                fitted, forecast = candidate_fitted, candidate_forecast
                best_order, best_seasonal_order = order, seasonal_order
                fallback_used = rank > 0
                break

        if forecast is None:
            best_order, best_seasonal_order = (1, 1, 0), (0, 0, 0, 0)
            fitted = fit_sarimax(SARIMAX, train_series, best_order, best_seasonal_order, exog=train_exog)
            forecast = fitted.forecast(steps=len(test_series), exog=test_exog)
            fallback_used = True

        current_metrics = regression_metrics(test_series.values, forecast.values)
        current_metrics["order_fallback_used"] = fallback_used
        current_metrics["provincia"] = province
        current_metrics["n_test"] = int(len(test_series))
        # Punto #6 del docente: un RMSE con pocas observaciones de prueba es inestable.
        # Se conserva el resultado pero se marca como exploratorio, no concluyente.
        current_metrics["exploratorio"] = bool(len(test_series) < MIN_TEST_OBS_FOR_PROVINCE_CONCLUSIONS)
        province_metrics.append(current_metrics)
        if not math.isnan(validation_rmse):
            validation_rmse_by_province.append(validation_rmse)

        full_series, full_exog = _prepare_series(group, bundle.target_column, exog_columns)
        final_fitted = fit_sarimax(SARIMAX, full_series, best_order, best_seasonal_order, exog=full_exog)

        province_key = str(province).lower().replace(" ", "_")
        model_path = directory / f"model_{province_key}.joblib"
        joblib.dump(final_fitted, model_path)
        model_files[str(province)] = str(model_path.relative_to(ROOT_DIR))
        selected_orders[str(province)] = {
            "order": best_order,
            "seasonal_order": best_seasonal_order,
            "validation_rmse_mean": validation_rmse,
            "tuning_results": tuning_results,
        }

        predictions = pd.DataFrame(
            {
                "fecha": test_series.index,
                # Se usa el valor de "producto" tal como aparece en el dataset (no
                # product.display_name) para que coincida byte a byte con XGBoost/LSTM y
                # la interseccion de la ventana comun de evaluacion (punto #3) funcione.
                "producto": test_part["producto"].iloc[0],
                "provincia": province,
                "y_true": test_series.values,
                "y_pred": forecast.values,
                "model_name": MODEL_NAME,
                "product_id": product.product_id,
                "feature_set": feature_set,
            }
        )
        prediction_parts.append(predictions)

    if not prediction_parts:
        raise ValueError(f"No hubo provincias con suficientes observaciones para {product.product_id} ({feature_set})")

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = regression_metrics(all_predictions["y_true"], all_predictions["y_pred"])
    metrics["validation_rmse_mean"] = (
        float(np.mean(validation_rmse_by_province)) if validation_rmse_by_province else math.nan
    )

    province_metrics_path = directory / "province_metrics.csv"
    pd.DataFrame(province_metrics).to_csv(province_metrics_path, index=False, encoding="utf-8-sig")

    metadata = {
        "product_id": product.product_id,
        "product_name": product.display_name,
        "model_name": MODEL_NAME,
        "feature_set": feature_set,
        "model_files": model_files,
        "target_column": bundle.target_column,
        "exog_columns": exog_columns,
        "holdout_test_ratio": HOLDOUT_TEST_RATIO,
        "order_grid": ORDER_GRID,
        "seasonal_order_grid": SEASONAL_ORDER_GRID,
        "selection_strategy": selection_strategy,
        "selected_orders_by_province": selected_orders,
        "min_observations": MIN_OBSERVATIONS,
        "min_test_obs_for_province_conclusions": MIN_TEST_OBS_FOR_PROVINCE_CONCLUSIONS,
        "validation_rmse_mean": metrics["validation_rmse_mean"],
        "province_metrics_file": str(province_metrics_path.relative_to(ROOT_DIR)),
    }
    write_json(directory / "config.json", metadata)
    write_model_result(product.product_id, MODEL_NAME, feature_set, metrics, metadata, all_predictions)
    return {"product_id": product.product_id, "feature_set": feature_set, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena ARIMA/SARIMAX por producto y provincia.")
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
