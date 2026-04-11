from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    add_monthly_date,
    drop_unnamed_columns,
    most_common_value,
    normalize_columns,
    normalize_key,
    normalize_series_text,
    read_csv_with_fallback,
    read_text_with_fallback,
    to_numeric,
)
from config import AGRO_FILE, IBC_FILE, IPC_FILE, IPM_FILE, IPPN_FILE, MARKETS_FILE, PRODUCTOR_FILE


def _prepare_common_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_columns(df)
    prepared = drop_unnamed_columns(prepared)
    prepared = add_monthly_date(prepared)
    prepared["provincia"] = normalize_series_text(prepared.get("provincia", pd.Series(dtype="string")), title_case=True)
    prepared["producto"] = normalize_series_text(prepared.get("producto", pd.Series(dtype="string")))
    if "provincia" in prepared.columns:
        prepared["provincia_key"] = prepared["provincia"].map(normalize_key)
    if "producto" in prepared.columns:
        prepared["producto_key"] = prepared["producto"].map(normalize_key)
    return prepared


def load_markets_product(product_key: str) -> pd.DataFrame:
    df = _prepare_common_frame(read_csv_with_fallback(MARKETS_FILE))
    df = df.loc[df["producto_key"] == product_key].copy()
    df["precio_promedio_usdkg"] = to_numeric(df["precio_promedio_usdkg"])
    df["promedio_de_precio_usd"] = to_numeric(df["promedio_de_precio_usd"])

    grouped = (
        df.groupby(["fecha", "provincia_key"], as_index=False)
        .agg(
            provincia=("provincia", most_common_value),
            producto=("producto", most_common_value),
            target_precio_mercado_usdkg=("precio_promedio_usdkg", "mean"),
            target_precio_mercado_usd=("promedio_de_precio_usd", "mean"),
            mercados_observaciones=("producto", "size"),
            mercados_distintos=("mercado", "nunique"),
            tipos_mercado_distintos=("tipo_mercado", "nunique"),
        )
        .sort_values(["provincia", "fecha"])
        .reset_index(drop=True)
    )
    return grouped


def load_productor_product(product_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _prepare_common_frame(read_csv_with_fallback(PRODUCTOR_FILE))
    df = df.loc[df["producto_key"] == product_key].copy()
    df["precio_promedio_usdkg"] = to_numeric(df["precio_promedio_usdkg"])

    provincial = (
        df.groupby(["fecha", "provincia_key"], as_index=False)
        .agg(
            provincia=("provincia", most_common_value),
            producto=("producto", most_common_value),
            precio_productor_provincia_usdkg=("precio_promedio_usdkg", "mean"),
            productor_observaciones_provincia=("producto", "size"),
        )
        .sort_values(["provincia", "fecha"])
        .reset_index(drop=True)
    )

    national = (
        df.groupby("fecha", as_index=False)
        .agg(
            precio_productor_nacional_usdkg=("precio_promedio_usdkg", "mean"),
            productor_observaciones_nacional=("producto", "size"),
        )
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    return provincial, national


def load_agro_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _prepare_common_frame(read_csv_with_fallback(AGRO_FILE))
    df["tipo_insumo_key"] = normalize_series_text(df["tipo_insumo"]).map(normalize_key)
    df = df.loc[df["tipo_insumo_key"] == "fertilizantes"].copy()
    df["promedio_de_precio"] = to_numeric(df["promedio_de_precio"])
    df["ingrediente_activo"] = normalize_series_text(df["ingrediente_activo"])

    provincial = (
        df.groupby(["fecha", "provincia_key"], as_index=False)
        .agg(
            provincia=("provincia", most_common_value),
            fertilizantes_precio_promedio_provincia=("promedio_de_precio", "mean"),
            fertilizantes_precio_mediano_provincia=("promedio_de_precio", "median"),
            fertilizantes_registros_provincia=("promedio_de_precio", "size"),
            fertilizantes_ingredientes_activos_provincia=("ingrediente_activo", "nunique"),
        )
        .sort_values(["provincia", "fecha"])
        .reset_index(drop=True)
    )

    national = (
        df.groupby("fecha", as_index=False)
        .agg(
            fertilizantes_precio_promedio_nacional=("promedio_de_precio", "mean"),
            fertilizantes_precio_mediano_nacional=("promedio_de_precio", "median"),
            fertilizantes_registros_nacional=("promedio_de_precio", "size"),
        )
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    return provincial, national


def load_ipc_features() -> pd.DataFrame:
    df = _prepare_common_frame(read_csv_with_fallback(IPC_FILE))
    numeric_columns = [
        "ipc_alimentos_y_bebidas_no_alcoholicas",
        "inflacion_mensual",
        "inflacion_anual",
        "inflacion_acumulada",
    ]
    for column in numeric_columns:
        df[column] = to_numeric(df[column])

    return (
        df[["fecha"] + numeric_columns]
        .rename(columns={"ipc_alimentos_y_bebidas_no_alcoholicas": "ipc_alimentos"})
        .sort_values("fecha")
        .reset_index(drop=True)
    )


def _find_header_row(path: Path, expected_prefix: str) -> int:
    for index, line in enumerate(read_text_with_fallback(path).splitlines()):
        if normalize_key(line).startswith(expected_prefix):
            return index
    raise ValueError(f"No se encontro la fila de encabezado esperada en {path.name}")


def load_sector_index(file_path: Path, value_column: str) -> pd.DataFrame:
    skiprows = 0
    if file_path == IBC_FILE:
        skiprows = _find_header_row(file_path, "ano,mes,ibc")

    df = normalize_columns(read_csv_with_fallback(file_path, skiprows=skiprows))
    df = drop_unnamed_columns(df)
    df = add_monthly_date(df)
    df[value_column] = to_numeric(df[value_column])

    return df[["fecha", value_column]].sort_values("fecha").reset_index(drop=True)


def load_macro_features() -> pd.DataFrame:
    ipc = load_ipc_features()
    ibc = load_sector_index(IBC_FILE, "ibc")
    ipm = load_sector_index(IPM_FILE, "ipm")
    ipp_n = load_sector_index(IPPN_FILE, "ipp_n")

    macro = ipc.merge(ibc, on="fecha", how="left")
    macro = macro.merge(ipm, on="fecha", how="left")
    macro = macro.merge(ipp_n, on="fecha", how="left")
    return macro.sort_values("fecha").reset_index(drop=True)
