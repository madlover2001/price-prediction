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
from training.common.horizon import CALENDAR_FEATURES, TRUE_EXOG_BASE_COLUMNS, build_future_exog_frame
from training.common.metrics import regression_metrics
from training.common.registry import artifact_dir, selected_products, write_json, write_model_result
from training.common.splits import apply_holdout_cutoffs, compute_holdout_cutoffs, expanding_validation_windows

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


def _prepare_full_series(group: pd.DataFrame, target_column: str, exog_columns: list[str]):
    """Serie mensual continua de TODA la historia de la provincia (no de un tramo ya
    recortado). Rellenar huecos de calendario con contexto pasado completo evita que un
    tramo recortado (train/test/ventana) quede con un hueco inicial que ffill no pueda
    resolver, y evita cualquier fuga hacia el pasado (solo ffill, nunca interpolate/bfill
    -- ver docs/correcciones_docente.md, punto C.1 de la revision del companero)."""
    ordered = group.sort_values("fecha").set_index("fecha").asfreq("MS")
    series = ordered[target_column].ffill()
    exog = ordered[exog_columns].ffill() if exog_columns else None
    return series, exog


def _is_stable_forecast(forecast_values, reference_values, max_multiple: float = 5.0) -> bool:
    """Con enforce_stationarity=False, algunos ordenes del grid producen pronosticos que
    divergen numericamente (raices AR/estacionales fuera del circulo unitario). Se
    descartan aqui en vez de dejarlos contaminar la seleccion de orden o las metricas
    finales con RMSE de escala irreal (ver docs/correcciones_docente.md, nota tecnica).

    `max_multiple` se endurecio de 25x a 5x, y se agrego una cota de no-negatividad: con
    las ventanas compartidas de C.3, algunas combinaciones provincia-ventana producian
    pronosticos moderadamente divergentes (precios negativos o 3-4x el rango real) que
    25x no alcanzaba a descartar y contaminaban common_validation_window.csv con RMSE de
    escala irreal (ver docs/correcciones_docente.md)."""
    forecast_values = np.asarray(forecast_values, dtype=float)
    if forecast_values.size == 0 or not np.all(np.isfinite(forecast_values)):
        return False
    if np.any(forecast_values < 0):
        # Los precios de mercado (USD/kg) nunca son negativos; un pronostico negativo es
        # en si mismo evidencia de divergencia numerica del modelo.
        return False
    reference_values = np.asarray(reference_values, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    if reference_values.size == 0:
        return True
    reference_scale = max(float(np.max(np.abs(reference_values))), 1e-6)
    return bool(np.all(forecast_values <= max_multiple * reference_scale))


def _window_bounds_from_pooled(pooled_windows) -> list[tuple]:
    """Convierte las ventanas expansivas (calculadas una sola vez, agrupando todas las
    provincias) en simples limites de fecha (train_end, val_start, val_end). Estos limites
    son los mismos para las 3 familias de modelo -- corrige el punto C.3 de la revision
    del companero: antes SARIMAX recalculaba ventanas por provincia con cortes propios,
    distintos a los de XGBoost/LSTM."""
    bounds = []
    for train_window, val_window in pooled_windows:
        if train_window.empty or val_window.empty:
            continue
        bounds.append((train_window["fecha"].max(), val_window["fecha"].min(), val_window["fecha"].max()))
    return bounds


def select_best_order(
    SARIMAX,
    full_series: pd.Series,
    full_exog: pd.DataFrame | None,
    window_bounds: list[tuple],
) -> tuple[tuple, tuple, float, list[dict], list[dict]]:
    tuning_results = []

    for order in ORDER_GRID:
        for seasonal_order in SEASONAL_ORDER_GRID:
            window_rmses = []
            for train_end, val_start, val_end in window_bounds:
                fit_series = full_series.loc[:train_end]
                val_series = full_series.loc[val_start:val_end]
                if fit_series.empty or val_series.empty:
                    window_rmses.append(math.inf)
                    continue
                fit_exog = full_exog.loc[:train_end] if full_exog is not None else None
                val_exog = full_exog.loc[val_start:val_end] if full_exog is not None else None
                try:
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
                    "n_windows": len(window_bounds),
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


def _oof_and_operational(
    SARIMAX,
    full_series: pd.Series,
    full_exog: pd.DataFrame | None,
    window_bounds: list[tuple],
    order: tuple,
    seasonal_order: tuple,
    exog_columns: list[str],
) -> tuple[pd.DataFrame, list[tuple]]:
    """Con el orden ya elegido: genera predicciones OOF (una por cada mes de cada ventana
    de validacion; las ventanas de validacion son disjuntas en fecha, asi que la
    concatenacion cubre el 80% de desarrollo sin solapes -- insumo para la ventana comun
    de validacion entre familias, punto C.3) y, sobre la ULTIMA ventana, un pronostico
    operacional h=1..3 con exogenas propagadas al ultimo valor conocido (mismo supuesto
    de carry-forward que training/common/horizon.py) -- insumo para el criterio de
    seleccion de modelo, punto C.6/C.7."""
    oof_rows = []
    operational_errors: list[tuple] = []

    for window_index, (train_end, val_start, val_end) in enumerate(window_bounds):
        fit_series = full_series.loc[:train_end]
        val_series = full_series.loc[val_start:val_end]
        if fit_series.empty or val_series.empty:
            continue
        fit_exog = full_exog.loc[:train_end] if full_exog is not None else None
        val_exog = full_exog.loc[val_start:val_end] if full_exog is not None else None
        try:
            fitted = fit_sarimax(SARIMAX, fit_series, order, seasonal_order, exog=fit_exog)
            forecast = fitted.forecast(steps=len(val_series), exog=val_exog)
        except Exception:
            continue
        if not _is_stable_forecast(forecast.values, fit_series.values):
            continue

        for fecha, y_true, y_pred in zip(val_series.index, val_series.values, forecast.values):
            oof_rows.append({"fecha": fecha, "y_true": float(y_true), "y_pred": float(y_pred)})

        if window_index == len(window_bounds) - 1:
            max_h = min(3, len(val_series))
            if max_h > 0:
                future_exog = None
                if exog_columns and fit_exog is not None and not fit_exog.empty:
                    future_exog = build_future_exog_frame(fit_exog.iloc[-1], train_end, max_h, exog_columns)
                try:
                    operational_forecast = fitted.forecast(steps=max_h, exog=future_exog)
                    for h in range(1, max_h + 1):
                        operational_errors.append(
                            (h, float(val_series.iloc[h - 1]), float(operational_forecast.iloc[h - 1]))
                        )
                except Exception:
                    pass

    return pd.DataFrame(oof_rows), operational_errors


def train_product(product_id: str, feature_set: str) -> dict:
    SARIMAX = _import_sarimax()
    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product, feature_set=feature_set)
    directory = artifact_dir(product.product_id, MODEL_NAME, feature_set)

    exog_columns = EXOG_COLUMNS if feature_set == "full" else []
    cutoffs = compute_holdout_cutoffs(bundle.data, test_ratio=HOLDOUT_TEST_RATIO)
    # "base" y "full" seleccionan su orden SIEMPRE mediante la misma busqueda en grid
    # validada por ventanas expansivas (nunca via overrides manuales), y ambos usan las
    # MISMAS ventanas (por fecha, agrupando todas las provincias) que XGBoost/LSTM.
    selection_strategy = "validation_grid_search"

    train_df, _ = apply_holdout_cutoffs(bundle.data, cutoffs)
    pooled_windows = expanding_validation_windows(train_df)
    window_bounds = _window_bounds_from_pooled(pooled_windows)

    province_metrics = []
    prediction_parts = []
    oof_parts = []
    operational_errors_all: list[tuple] = []
    model_files = {}
    selected_orders = {}
    validation_rmse_by_province = []

    for province, group in bundle.data.sort_values(["provincia", "fecha"]).groupby("provincia", sort=False):
        if len(group) < MIN_OBSERVATIONS:
            continue
        cutoff = cutoffs.get(province)
        if cutoff is None:
            continue

        full_series, full_exog = _prepare_full_series(group, bundle.target_column, exog_columns)
        train_series = full_series.loc[full_series.index < cutoff]
        test_series = full_series.loc[full_series.index >= cutoff]
        train_exog = full_exog.loc[full_exog.index < cutoff] if full_exog is not None else None
        test_exog = full_exog.loc[full_exog.index >= cutoff] if full_exog is not None else None
        if train_series.empty or test_series.empty:
            continue

        # Ventanas globales, recortadas a las que caen enteramente antes del holdout de
        # ESTA provincia (una provincia con cutoff mas temprano que otras no debe validar
        # con fechas que para ella ya son parte del test).
        province_window_bounds = [bounds for bounds in window_bounds if bounds[2] < cutoff]

        best_order, best_seasonal_order, validation_rmse, tuning_results, ranked_candidates = select_best_order(
            SARIMAX, full_series, full_exog, province_window_bounds
        )

        # El orden ganador en validacion puede seguir siendo inestable sobre el horizonte
        # de test (mas largo que las ventanas de validacion). Se prueban los siguientes
        # candidatos rankeados hasta encontrar un pronostico numericamente estable; si
        # ninguno lo es, se cae a un orden conservador por defecto como ultimo recurso.
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

        oof_df, operational_errors = _oof_and_operational(
            SARIMAX, full_series, full_exog, province_window_bounds, best_order, best_seasonal_order, exog_columns
        )
        if not oof_df.empty:
            oof_df["producto"] = group["producto"].iloc[0]
            oof_df["provincia"] = province
            oof_parts.append(oof_df)
        operational_errors_all.extend(operational_errors)

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
                "producto": group["producto"].iloc[0],
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

    # Criterio operacional de seleccion (C.6/C.7): RMSE agregado de los pronosticos
    # h=1,2,3 con carry-forward de la ultima ventana de validacion de cada provincia,
    # nunca toca el test.
    if operational_errors_all:
        y_true_op = np.array([row[1] for row in operational_errors_all], dtype=float)
        y_pred_op = np.array([row[2] for row in operational_errors_all], dtype=float)
        metrics["operational_rmse_mean"] = float(np.sqrt(np.mean((y_true_op - y_pred_op) ** 2)))
    else:
        metrics["operational_rmse_mean"] = math.nan

    oof_predictions = (
        pd.concat(oof_parts, ignore_index=True)
        if oof_parts
        else pd.DataFrame(columns=["fecha", "y_true", "y_pred", "producto", "provincia"])
    )
    if not oof_predictions.empty:
        oof_predictions["model_name"] = MODEL_NAME
        oof_predictions["product_id"] = product.product_id
        oof_predictions["feature_set"] = feature_set
    validation_predictions_path = directory / "validation_predictions.csv"
    oof_predictions.to_csv(validation_predictions_path, index=False, encoding="utf-8-sig")

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
        "operational_rmse_mean": metrics["operational_rmse_mean"],
        "n_validation_windows": len(window_bounds),
        "province_metrics_file": str(province_metrics_path.relative_to(ROOT_DIR)),
        "validation_predictions_file": str(validation_predictions_path.relative_to(ROOT_DIR)),
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
