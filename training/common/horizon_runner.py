from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from training.common.config import FORECAST_HORIZONS
from training.common.data_loader import load_product_dataset
from training.common.horizon import build_future_exog_frame, evaluate_recursive_horizons
from training.common.metrics import regression_metrics
from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS, artifact_dir, read_json, write_json
from training.common.splits import apply_holdout_cutoffs, compute_holdout_cutoffs

# Orquesta la evaluacion de horizontes 1-3 meses (punto #5 del docente) para los 3
# modelos entrenados con feature_set="full" (la configuracion real del prototipo). Cada
# familia de modelo predice recursivamente usando solo historia real anterior a cada
# fecha de origen del holdout de test; las exogenas verdaderas futuras se propagan al
# ultimo valor conocido (ver training/common/horizon.py).

HORIZON_PREDICTIONS_DIR = MODEL_RESULTS_DIR / "horizon_predictions"


def _xgboost_predict_factory(product_id: str):
    directory = artifact_dir(product_id, "xgboost", "full")
    model_path = directory / "model.joblib"
    if not model_path.exists():
        return None

    model = joblib.load(model_path)
    metadata = read_json(directory / "training_metadata.json")
    feature_columns = metadata["feature_columns"]
    blend_feature = metadata.get("lag_blend_feature")
    blend_weight = metadata.get("lag_blend_weight", 0)

    def factory(_province):
        def predict_one_step(feature_row: dict, _working_history: pd.DataFrame) -> float:
            frame = pd.DataFrame([feature_row])[feature_columns]
            raw = float(model.predict(frame)[0])
            if blend_feature and blend_feature in feature_row:
                return (1 - blend_weight) * raw + blend_weight * feature_row[blend_feature]
            return raw

        return predict_one_step

    return factory


def _lstm_predict_factory(product_id: str):
    directory = artifact_dir(product_id, "lstm", "full")
    model_path = directory / "model.keras"
    if not model_path.exists():
        return None

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    feature_scaler = joblib.load(directory / "feature_scaler.joblib")
    target_scaler = joblib.load(directory / "target_scaler.joblib")
    metadata = read_json(directory / "training_metadata.json")
    feature_columns = metadata["feature_columns"]
    window_size = metadata["window_size"]

    def factory(_province):
        def predict_one_step(feature_row: dict, working_history: pd.DataFrame) -> float:
            context = working_history.tail(window_size - 1)
            window_df = pd.concat([context, pd.DataFrame([feature_row])], ignore_index=True)
            if len(window_df) < window_size:
                return float("nan")
            values = feature_scaler.transform(window_df[feature_columns])
            X = values.reshape(1, window_size, len(feature_columns))
            pred_scaled = model.predict(X, verbose=0).reshape(-1)[0]
            return float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])

        return predict_one_step

    return factory


def _sarimax_horizon_predictions(product_id: str, target_column: str, horizons: tuple) -> pd.DataFrame:
    directory = artifact_dir(product_id, "arima", "full")
    metadata_path = directory / "training_metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()

    from training.arima.train import _import_sarimax, _is_stable_forecast, _prepare_series, fit_sarimax

    metadata = read_json(metadata_path)
    exog_columns = metadata.get("exog_columns", [])
    orders_by_province = metadata.get("selected_orders_by_province", {})

    bundle = load_product_dataset(PRODUCTS[product_id], feature_set="full")
    cutoffs = compute_holdout_cutoffs(bundle.data)
    SARIMAX = _import_sarimax()
    max_horizon = max(horizons)
    rows = []

    for province, province_df in bundle.data.groupby("provincia", sort=False):
        order_info = orders_by_province.get(str(province))
        cutoff = cutoffs.get(province)
        if order_info is None or cutoff is None:
            continue
        order = tuple(order_info["order"])
        seasonal_order = tuple(order_info["seasonal_order"])

        province_df = province_df.sort_values("fecha").reset_index(drop=True)
        actuals = province_df.set_index("fecha")[target_column]
        origin_dates = sorted(province_df.loc[province_df["fecha"] >= cutoff, "fecha"].unique())

        for origin_date in origin_dates:
            history = province_df[province_df["fecha"] < origin_date]
            if history.empty:
                continue
            series, exog = _prepare_series(history, target_column, exog_columns)
            last_row = history.iloc[-1]
            last_date = history["fecha"].max()
            future_exog = (
                build_future_exog_frame(last_row, last_date, max_horizon, exog_columns) if exog_columns else None
            )

            # Mismo resguardo de estabilidad que training/arima/train.py: el orden
            # elegido en entrenamiento puede divergir al refitearse sobre un origen de
            # fecha distinto. Si diverge, se cae al orden conservador por defecto; si aun
            # asi diverge, se descarta ese origen en vez de contaminar horizon_metrics.csv
            # con RMSE de escala irreal.
            forecast = None
            for candidate_order, candidate_seasonal_order in ((order, seasonal_order), ((1, 1, 0), (0, 0, 0, 0))):
                try:
                    fitted = fit_sarimax(SARIMAX, series, candidate_order, candidate_seasonal_order, exog=exog)
                    candidate_forecast = fitted.forecast(steps=max_horizon, exog=future_exog)
                except Exception:
                    continue
                if _is_stable_forecast(candidate_forecast.values, series.values):
                    forecast = candidate_forecast
                    break
            if forecast is None:
                continue

            for step in range(1, max_horizon + 1):
                if step not in horizons:
                    continue
                target_date = last_date + pd.DateOffset(months=step)
                if target_date not in actuals.index:
                    continue
                rows.append(
                    {
                        "provincia": province,
                        "fecha_origen": origin_date,
                        "horizonte": step,
                        "fecha_objetivo": target_date,
                        "y_true": actuals.loc[target_date],
                        "y_pred": float(forecast.iloc[step - 1]),
                    }
                )

    return pd.DataFrame(rows)


def _run_model_horizons(model_name: str, product_id: str, horizons: tuple) -> pd.DataFrame:
    if model_name == "arima":
        bundle = load_product_dataset(PRODUCTS[product_id], feature_set="full")
        return _sarimax_horizon_predictions(product_id, bundle.target_column, horizons)

    factory = _xgboost_predict_factory(product_id) if model_name == "xgboost" else _lstm_predict_factory(product_id)
    if factory is None:
        return pd.DataFrame()

    bundle = load_product_dataset(PRODUCTS[product_id], feature_set="full")
    cutoffs = compute_holdout_cutoffs(bundle.data)
    _, test_df = apply_holdout_cutoffs(bundle.data, cutoffs)

    return evaluate_recursive_horizons(
        factory,
        bundle.data,
        test_df,
        bundle.feature_columns,
        bundle.target_column,
        horizons=horizons,
    )


def run_horizon_evaluation(horizons: tuple = FORECAST_HORIZONS) -> None:
    HORIZON_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for product_id, product in PRODUCTS.items():
        for model_name in ("xgboost", "lstm", "arima"):
            predictions = _run_model_horizons(model_name, product_id, horizons)
            if predictions.empty:
                continue

            predictions.to_csv(
                HORIZON_PREDICTIONS_DIR / f"{product_id}_{model_name}_full_horizon.csv",
                index=False,
                encoding="utf-8-sig",
            )

            for horizon, group in predictions.groupby("horizonte"):
                # LSTM no puede pronosticar cuando el historial disponible antes del
                # origen es menor que WINDOW_SIZE-1 (provincias que se incorporan tarde
                # al panel); esas filas quedan como NaN en el CSV para transparencia pero
                # se excluyen del calculo de metricas.
                group = group.dropna(subset=["y_pred"])
                if group.empty:
                    continue
                metrics = regression_metrics(group["y_true"], group["y_pred"])
                metric_rows.append(
                    {
                        "product_id": product_id,
                        "product_name": product.display_name,
                        "model_name": model_name,
                        "feature_set": "full",
                        "horizonte": int(horizon),
                        **metrics,
                    }
                )

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    horizon_metrics_path = MODEL_RESULTS_DIR / "horizon_metrics.csv"
    if metric_rows:
        pd.DataFrame(metric_rows).sort_values(["product_id", "model_name", "horizonte"]).to_csv(
            horizon_metrics_path, index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(horizon_metrics_path, index=False, encoding="utf-8-sig")
