from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from training.common.config import FORECAST_HORIZONS

# Ver docs/correcciones_docente.md, punto 5 ("horizontes de 1 a 3 meses").
#
# Estrategia para variables futuras desconocidas: las columnas de calendario son
# deterministas y se recalculan exactas para cada mes futuro. Las exogenas verdaderas
# (precio productor, fertilizantes, indicadores macro/sectoriales) no se conocen para
# meses futuros, asi que se propagan al ultimo valor observado (carry-forward),
# documentado explicitamente como supuesto/limitacion del prototipo. Los rezagos y
# ventanas moviles del propio target se recalculan de forma recursiva usando las
# predicciones que el mismo modelo va generando dentro del horizonte.

CALENDAR_FEATURES = ["mes_num", "trimestre", "mes_sin", "mes_cos"]

TRUE_EXOG_BASE_COLUMNS = [
    "mercados_observaciones",
    "mercados_distintos",
    "tipos_mercado_distintos",
    "precio_productor_provincia_usdkg",
    "precio_productor_nacional_usdkg",
    "precio_productor_usdkg_filled",
    "productor_missing_exact",
    "fertilizantes_precio_promedio_provincia",
    "fertilizantes_precio_promedio_nacional",
    "fertilizantes_precio_promedio_filled",
    "fertilizantes_registros_provincia",
    "fertilizantes_ingredientes_activos_provincia",
    "fertilizantes_missing_exact",
    "ipc_alimentos",
    "inflacion_mensual",
    "inflacion_anual",
    "inflacion_acumulada",
    "ibc",
    "ipm",
    "ipp_n",
]

# Cada rezago exogeno hereda el carry-forward de su columna base: si el valor
# contemporaneo se mantiene constante hacia el futuro, su rezago tambien.
EXOGENOUS_LAG_TO_BASE_COLUMN = {
    "productor_lag_1": "precio_productor_usdkg_filled",
    "productor_lag_3": "precio_productor_usdkg_filled",
    "fertilizantes_lag_1": "fertilizantes_precio_promedio_filled",
    "fertilizantes_lag_3": "fertilizantes_precio_promedio_filled",
    "ipc_alimentos_lag_1": "ipc_alimentos",
    "inflacion_mensual_lag_1": "inflacion_mensual",
    "ibc_lag_1": "ibc",
    "ipm_lag_1": "ipm",
    "ipp_n_lag_1": "ipp_n",
}

TARGET_LAG_STEPS = {
    "target_lag_1": 1,
    "target_lag_2": 2,
    "target_lag_3": 3,
    "target_lag_6": 6,
    "target_lag_12": 12,
}
TARGET_ROLLING_WINDOWS = {
    "target_rolling_mean_3": (3, "mean"),
    "target_rolling_std_3": (3, "std"),
    "target_rolling_mean_6": (6, "mean"),
    "target_rolling_std_6": (6, "std"),
}


def _target_derived_features(target_history: list[float]) -> dict:
    """Replica, para un unico paso futuro, la logica de rezagos/rolling de
    feature_engineering_productos/feature_builder.py::_add_group_lag_features.
    `target_history` es la serie real+predicha hasta el mes anterior al que se pronostica."""
    values = pd.Series(target_history, dtype=float)
    features = {}
    for name, step in TARGET_LAG_STEPS.items():
        features[name] = float(values.iloc[-step]) if len(values) >= step else np.nan
    for name, (window, stat) in TARGET_ROLLING_WINDOWS.items():
        tail = values.iloc[-window:] if len(values) >= window else values
        if tail.empty:
            features[name] = np.nan
        elif stat == "mean":
            features[name] = float(tail.mean())
        else:
            features[name] = float(tail.std()) if len(tail) > 1 else 0.0
    features["target_momentum_1_3"] = features["target_lag_1"] - features["target_lag_3"]
    features["target_momentum_1_6"] = features["target_lag_1"] - features["target_lag_6"]
    return features


def build_future_feature_row(
    last_known_row: pd.Series,
    target_history: list[float],
    future_date: pd.Timestamp,
    feature_columns: list[str],
) -> dict:
    row: dict = {
        "mes_num": future_date.month,
        "trimestre": (future_date.month - 1) // 3 + 1,
        "mes_sin": float(np.sin(2 * np.pi * future_date.month / 12)),
        "mes_cos": float(np.cos(2 * np.pi * future_date.month / 12)),
    }
    for column in TRUE_EXOG_BASE_COLUMNS:
        if column in last_known_row.index:
            row[column] = last_known_row[column]
    for lag_name, base_column in EXOGENOUS_LAG_TO_BASE_COLUMN.items():
        if base_column in last_known_row.index:
            row[lag_name] = last_known_row[base_column]
    row.update(_target_derived_features(target_history))
    if "provincia_id" in last_known_row.index:
        row["provincia_id"] = last_known_row["provincia_id"]
    return {column: row[column] for column in feature_columns if column in row}


def build_future_exog_frame(
    last_known_row: pd.Series,
    last_date: pd.Timestamp,
    horizon: int,
    exog_columns: list[str],
) -> pd.DataFrame:
    """Exogenas futuras para SARIMAX (`exog=`): statsmodels ya resuelve internamente la
    dependencia recursiva del propio target vía el orden AR/estacional, asi que aqui solo
    hace falta el calendario exacto y el carry-forward de las exogenas verdaderas."""
    rows = []
    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)
        row = {
            "fecha": future_date,
            "mes_num": future_date.month,
            "trimestre": (future_date.month - 1) // 3 + 1,
            "mes_sin": float(np.sin(2 * np.pi * future_date.month / 12)),
            "mes_cos": float(np.cos(2 * np.pi * future_date.month / 12)),
        }
        for column in TRUE_EXOG_BASE_COLUMNS:
            if column in last_known_row.index:
                row[column] = last_known_row[column]
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("fecha")
    return frame[[column for column in exog_columns if column in frame.columns]]


def recursive_forecast(
    predict_one_step_fn: Callable[[dict, pd.DataFrame], float],
    history_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    target_column: str,
    date_col: str = "fecha",
) -> pd.DataFrame:
    """Pronostico recursivo h=1..horizon para una sola provincia.

    `predict_one_step_fn(feature_row, working_history)` predice un paso adelante: recibe
    el diccionario de features del mes a predecir y el historial acumulado (real +
    predicciones previas del propio horizonte), y cada modelo decide que necesita de ahi
    (XGBoost usa solo `feature_row`; LSTM reconstruye la ventana de `WINDOW_SIZE` meses
    desde `working_history`).
    """
    working_history = history_df.sort_values(date_col).reset_index(drop=True)
    target_history = working_history[target_column].tolist()
    last_row = working_history.iloc[-1]
    last_date = last_row[date_col]

    rows = []
    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)
        feature_row = build_future_feature_row(last_row, target_history, future_date, feature_columns)
        prediction = float(predict_one_step_fn(feature_row, working_history))

        new_row = dict(feature_row)
        new_row[date_col] = future_date
        new_row[target_column] = prediction
        working_history = pd.concat([working_history, pd.DataFrame([new_row])], ignore_index=True)
        target_history.append(prediction)
        rows.append({date_col: future_date, "horizonte": step, "y_pred": prediction})

    return pd.DataFrame(rows)


def evaluate_recursive_horizons(
    predict_one_step_fn_factory: Callable[[str], Callable[[dict, pd.DataFrame], float] | None],
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizons: tuple = FORECAST_HORIZONS,
    group_col: str = "provincia",
    date_col: str = "fecha",
) -> pd.DataFrame:
    """Evalua horizontes 1..max(horizons) usando como origen cada fecha del holdout de
    test. La historia usada para pronosticar cada origen es siempre estrictamente anterior
    a esa fecha (nunca incluye filas de test posteriores), y el valor real se compara solo
    cuando existe en el dataset completo."""
    max_horizon = max(horizons)
    rows = []
    full_df = full_df.sort_values([group_col, date_col])

    for province, province_full in full_df.groupby(group_col, sort=False):
        province_full = province_full.reset_index(drop=True)
        predict_one_step_fn = predict_one_step_fn_factory(province)
        if predict_one_step_fn is None:
            continue

        origin_dates = sorted(test_df.loc[test_df[group_col] == province, date_col].unique())
        actuals = province_full.set_index(date_col)[target_column]

        for origin_date in origin_dates:
            history = province_full[province_full[date_col] < origin_date]
            if history.empty:
                continue
            forecast = recursive_forecast(
                predict_one_step_fn, history, feature_columns, max_horizon, target_column, date_col
            )
            for _, forecast_row in forecast.iterrows():
                step = int(forecast_row["horizonte"])
                if step not in horizons:
                    continue
                target_date = forecast_row[date_col]
                if target_date not in actuals.index:
                    continue
                rows.append(
                    {
                        "provincia": province,
                        "fecha_origen": origin_date,
                        "horizonte": step,
                        "fecha_objetivo": target_date,
                        "y_true": actuals.loc[target_date],
                        "y_pred": forecast_row["y_pred"],
                    }
                )

    return pd.DataFrame(rows)
