from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_directory, write_json
from config import COMMON_START_DATE, PRODUCT_CONFIGS
from data_sources import (
    load_agro_features,
    load_macro_features,
    load_markets_product,
    load_productor_product,
)


TARGET_COLUMN = "target_precio_mercado_usdkg"

CALENDAR_FEATURES = ["mes_num", "trimestre", "mes_sin", "mes_cos"]

TARGET_LAG_FEATURES = [
    "target_lag_1",
    "target_lag_2",
    "target_lag_3",
    "target_lag_6",
    "target_lag_12",
    "target_rolling_mean_3",
    "target_rolling_std_3",
    "target_rolling_mean_6",
    "target_rolling_std_6",
    "target_momentum_1_3",
    "target_momentum_1_6",
]

# Campos de contexto de mercado (conteos de la propia fuente mercados): no son ninguna de
# las 4 categorias que la hipotesis define como exogenas (precio productor, fertilizantes,
# IPC/inflacion, indices sectoriales). Se quedan documentados en el dataset y en
# feature_groups.market_context, pero NO entran a full_model_features -- corrige el punto
# C.4 de la revision del companero: antes "full" mezclaba exogenas de la hipotesis con
# contexto de mercado, dejando la ablacion menos limpia de lo que la hipotesis promete.
MARKET_CONTEXT_FEATURES = [
    "mercados_observaciones",
    "mercados_distintos",
    "tipos_mercado_distintos",
]

EXOGENOUS_TRUE_FEATURES = [
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

EXOGENOUS_LAG_FEATURES = [
    "productor_lag_1",
    "productor_lag_3",
    "fertilizantes_lag_1",
    "fertilizantes_lag_3",
    "ipc_alimentos_lag_1",
    "inflacion_mensual_lag_1",
    "ibc_lag_1",
    "ipm_lag_1",
    "ipp_n_lag_1",
]

# "base" = solo historia propia del precio objetivo + calendario + identificador espacial.
# "full" = base + variables exogenas verdaderas (productor, fertilizantes, macro/sectoriales)
# contemporaneas y rezagadas. Esta separacion sostiene los experimentos de ablacion que miden
# el aporte real de las exogenas (ver docs/correcciones_docente.md, punto 1).
BASE_MODEL_FEATURES = ["provincia_id"] + CALENDAR_FEATURES + TARGET_LAG_FEATURES
FULL_MODEL_FEATURES = BASE_MODEL_FEATURES + EXOGENOUS_TRUE_FEATURES + EXOGENOUS_LAG_FEATURES

RECOMMENDED_FEATURES = FULL_MODEL_FEATURES
LAG_FEATURES = TARGET_LAG_FEATURES + EXOGENOUS_LAG_FEATURES


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["anio"] = enriched["fecha"].dt.year
    enriched["mes_num"] = enriched["fecha"].dt.month
    enriched["trimestre"] = enriched["fecha"].dt.quarter
    enriched["mes_sin"] = np.sin(2 * np.pi * enriched["mes_num"] / 12)
    enriched["mes_cos"] = np.cos(2 * np.pi * enriched["mes_num"] / 12)
    return enriched


def _fill_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()

    filled["productor_missing_exact"] = filled["precio_productor_provincia_usdkg"].isna().astype(int)
    filled["precio_productor_usdkg_filled"] = filled["precio_productor_provincia_usdkg"].fillna(
        filled["precio_productor_nacional_usdkg"]
    )
    filled["precio_productor_usdkg_filled"] = (
        filled.groupby("provincia_key")["precio_productor_usdkg_filled"].transform(lambda s: s.ffill())
    )
    filled["precio_productor_provincia_usdkg"] = filled["precio_productor_provincia_usdkg"].fillna(
        filled["precio_productor_usdkg_filled"]
    )
    filled["precio_productor_nacional_usdkg"] = filled["precio_productor_nacional_usdkg"].fillna(
        filled["precio_productor_usdkg_filled"]
    )
    filled["productor_observaciones_provincia"] = filled["productor_observaciones_provincia"].fillna(0)
    filled["productor_observaciones_nacional"] = filled["productor_observaciones_nacional"].fillna(0)

    filled["fertilizantes_missing_exact"] = filled["fertilizantes_precio_promedio_provincia"].isna().astype(int)
    filled["fertilizantes_precio_promedio_filled"] = filled["fertilizantes_precio_promedio_provincia"].fillna(
        filled["fertilizantes_precio_promedio_nacional"]
    )
    filled["fertilizantes_precio_promedio_filled"] = filled.groupby("provincia_key")[
        "fertilizantes_precio_promedio_filled"
    ].transform(lambda s: s.ffill())
    filled["fertilizantes_precio_promedio_provincia"] = filled["fertilizantes_precio_promedio_provincia"].fillna(
        filled["fertilizantes_precio_promedio_filled"]
    )
    filled["fertilizantes_precio_promedio_nacional"] = filled["fertilizantes_precio_promedio_nacional"].fillna(
        filled["fertilizantes_precio_promedio_filled"]
    )
    filled["fertilizantes_precio_mediano_provincia"] = filled["fertilizantes_precio_mediano_provincia"].fillna(
        filled["fertilizantes_precio_mediano_nacional"]
    )
    filled["fertilizantes_precio_mediano_nacional"] = filled["fertilizantes_precio_mediano_nacional"].fillna(
        filled["fertilizantes_precio_mediano_provincia"]
    )
    filled["fertilizantes_registros_provincia"] = filled["fertilizantes_registros_provincia"].fillna(0)
    filled["fertilizantes_registros_nacional"] = filled["fertilizantes_registros_nacional"].fillna(0)
    filled["fertilizantes_ingredientes_activos_provincia"] = filled[
        "fertilizantes_ingredientes_activos_provincia"
    ].fillna(0)

    national_fill_columns = [
        "ipc_alimentos",
        "inflacion_mensual",
        "inflacion_anual",
        "inflacion_acumulada",
        "ibc",
        "ipm",
        "ipp_n",
    ]
    # Bandera para el punto C.9 (cuantificar el preprocesamiento): cuantas filas tenian
    # algun indicador macro/sectorial faltante ANTES de completar, capturada antes del
    # ffill de abajo.
    filled["macro_missing_any"] = filled[national_fill_columns].isna().any(axis=1).astype(int)

    # ffill only: bfill would fill early gaps with a later (future) value, which is a
    # look-ahead leak relative to the row's own date. Rows still missing after ffill
    # (no observation yet exists) are dropped by the dropna in build_product_dataset.
    for column in national_fill_columns:
        filled[column] = filled[column].ffill()

    return filled


def _add_group_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lagged = df.sort_values(["provincia_key", "fecha"]).copy()
    by_province = lagged.groupby("provincia_key", group_keys=False)

    lagged["target_lag_1"] = by_province[TARGET_COLUMN].shift(1)
    lagged["target_lag_2"] = by_province[TARGET_COLUMN].shift(2)
    lagged["target_lag_3"] = by_province[TARGET_COLUMN].shift(3)
    lagged["target_lag_6"] = by_province[TARGET_COLUMN].shift(6)
    lagged["target_lag_12"] = by_province[TARGET_COLUMN].shift(12)

    lagged["target_rolling_mean_3"] = by_province[TARGET_COLUMN].shift(1).rolling(3).mean()
    lagged["target_rolling_std_3"] = by_province[TARGET_COLUMN].shift(1).rolling(3).std()
    lagged["target_rolling_mean_6"] = by_province[TARGET_COLUMN].shift(1).rolling(6).mean()
    lagged["target_rolling_std_6"] = by_province[TARGET_COLUMN].shift(1).rolling(6).std()

    lagged["target_momentum_1_3"] = lagged["target_lag_1"] - lagged["target_lag_3"]
    lagged["target_momentum_1_6"] = lagged["target_lag_1"] - lagged["target_lag_6"]

    lagged["productor_lag_1"] = by_province["precio_productor_usdkg_filled"].shift(1)
    lagged["productor_lag_3"] = by_province["precio_productor_usdkg_filled"].shift(3)
    lagged["fertilizantes_lag_1"] = by_province["fertilizantes_precio_promedio_filled"].shift(1)
    lagged["fertilizantes_lag_3"] = by_province["fertilizantes_precio_promedio_filled"].shift(3)
    lagged["ipc_alimentos_lag_1"] = by_province["ipc_alimentos"].shift(1)
    lagged["inflacion_mensual_lag_1"] = by_province["inflacion_mensual"].shift(1)
    lagged["ibc_lag_1"] = by_province["ibc"].shift(1)
    lagged["ipm_lag_1"] = by_province["ipm"].shift(1)
    lagged["ipp_n_lag_1"] = by_province["ipp_n"].shift(1)

    return lagged


def _build_preprocessing_summary(registros_iniciales: int, final_df: pd.DataFrame) -> dict:
    """Cuantificacion del preprocesamiento (punto C.9 de la revision del companero): el
    tutor pidio porcentaje eliminado/imputado/transformado, no solo una descripcion del
    procedimiento."""
    registros_finales = int(len(final_df))
    eliminados = int(registros_iniciales - registros_finales)
    productor_imputado = int(final_df["productor_missing_exact"].sum()) if not final_df.empty else 0
    fertilizantes_imputado = int(final_df["fertilizantes_missing_exact"].sum()) if not final_df.empty else 0
    macro_completado = int(final_df["macro_missing_any"].sum()) if not final_df.empty else 0

    def _pct(value: int, total: int) -> float:
        return round(value / total * 100, 2) if total else 0.0

    return {
        "registros_iniciales": int(registros_iniciales),
        "registros_eliminados": eliminados,
        "pct_eliminado": _pct(eliminados, registros_iniciales),
        "productor_imputado": productor_imputado,
        "pct_productor_imputado": _pct(productor_imputado, registros_finales),
        "fertilizantes_imputado": fertilizantes_imputado,
        "pct_fertilizantes_imputado": _pct(fertilizantes_imputado, registros_finales),
        "macro_completado": macro_completado,
        "pct_macro_completado": _pct(macro_completado, registros_finales),
        "registros_finales": registros_finales,
    }


def _build_metadata(
    product_config: dict,
    base_df: pd.DataFrame,
    final_df: pd.DataFrame,
    shared_provinces: list[str],
    productor_exact_matches: int,
    registros_iniciales: int,
) -> dict:
    missing_counts = final_df.isna().sum()
    missing_counts = {key: int(value) for key, value in missing_counts.items() if int(value) > 0}
    preprocessing_summary = _build_preprocessing_summary(registros_iniciales, final_df)

    return {
        "product_name": product_config["product_name"],
        "product_key": product_config["product_key"],
        "target_column": TARGET_COLUMN,
        "entity_columns": ["provincia"],
        "date_column": "fecha",
        "source_files": {
            "mercados": "outputs/precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_Mercados12-25.csv",
            "productor": "outputs/precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv",
            "agro": "outputs/precios-agroquimicos-fertilizantes.xlsx_Hoja1.csv",
            "ipc": "outputs/ipc-alimentos-inflacion.xlsx_Inflación.csv",
            "ibc": "outputs/indices-sector.xlsx_IBC.csv",
            "ipm": "outputs/indices-sector.xlsx_IPM.csv",
            "ipp_n": "outputs/indices-sector.xlsx_IPP-N.csv",
        },
        "rows_before_lag_drop": int(len(base_df)),
        "rows_final_dataset": int(len(final_df)),
        "date_min": final_df["fecha"].min().strftime("%Y-%m-%d") if not final_df.empty else None,
        "date_max": final_df["fecha"].max().strftime("%Y-%m-%d") if not final_df.empty else None,
        "shared_provinces_count": len(shared_provinces),
        "shared_provinces": shared_provinces,
        "productor_exact_match_rows": int(productor_exact_matches),
        "productor_exact_match_pct": round(productor_exact_matches / len(base_df), 4) if len(base_df) else 0,
        "feature_groups": {
            "identifiers": ["fecha", "producto", "provincia", "provincia_id", "anio", "mes_num", "trimestre"],
            "target": [TARGET_COLUMN, "target_precio_mercado_usd"],
            "market_context": MARKET_CONTEXT_FEATURES,
            "producer_features": [
                "precio_productor_provincia_usdkg",
                "precio_productor_nacional_usdkg",
                "precio_productor_usdkg_filled",
                "productor_observaciones_provincia",
                "productor_observaciones_nacional",
                "productor_missing_exact",
            ],
            "agro_features": [
                "fertilizantes_precio_promedio_provincia",
                "fertilizantes_precio_promedio_nacional",
                "fertilizantes_precio_promedio_filled",
                "fertilizantes_precio_mediano_provincia",
                "fertilizantes_precio_mediano_nacional",
                "fertilizantes_registros_provincia",
                "fertilizantes_registros_nacional",
                "fertilizantes_ingredientes_activos_provincia",
                "fertilizantes_missing_exact",
            ],
            "macro_features": [
                "ipc_alimentos",
                "inflacion_mensual",
                "inflacion_anual",
                "inflacion_acumulada",
                "ibc",
                "ipm",
                "ipp_n",
            ],
            "temporal_features": ["mes_sin", "mes_cos"],
            "lag_features": LAG_FEATURES,
        },
        "recommended_model_features": RECOMMENDED_FEATURES,
        "base_model_features": BASE_MODEL_FEATURES,
        "full_model_features": FULL_MODEL_FEATURES,
        "missing_values_after_export": missing_counts,
        "preprocessing_summary": preprocessing_summary,
    }


def build_product_dataset(product_id: str, output_dir: Path | None = None) -> tuple[Path, Path]:
    if product_id not in PRODUCT_CONFIGS:
        raise ValueError(f"Producto no configurado: {product_id}")

    product_config = PRODUCT_CONFIGS[product_id]
    final_output_dir = output_dir or product_config["output_dir"]
    ensure_directory(final_output_dir)

    target_df = load_markets_product(product_config["product_key"])
    productor_prov_df, productor_nat_df = load_productor_product(product_config["product_key"])
    agro_prov_df, agro_nat_df = load_agro_features()
    macro_df = load_macro_features()

    shared_province_keys = sorted(set(target_df["provincia_key"]).intersection(set(productor_prov_df["provincia_key"])))
    target_df = target_df.loc[target_df["provincia_key"].isin(shared_province_keys)].copy()
    target_df = target_df.loc[target_df["fecha"] >= COMMON_START_DATE].copy()
    # Punto de partida para la cuantificacion del preprocesamiento (C.9): registros de
    # mercados ya acotados a provincias compartidas con productor y a la fecha comun, que
    # es la base real sobre la que se construye el dataset (los merges de abajo son "left"
    # y preservan este conteo; lo que reduce el numero de filas es el dropna final).
    registros_iniciales = int(len(target_df))

    province_name_map = (
        target_df[["provincia_key", "provincia"]]
        .drop_duplicates()
        .sort_values(["provincia"])
        .reset_index(drop=True)
    )
    province_name_map["provincia_id"] = range(len(province_name_map))

    dataset = target_df.merge(
        productor_prov_df[
            ["fecha", "provincia_key", "precio_productor_provincia_usdkg", "productor_observaciones_provincia"]
        ],
        on=["fecha", "provincia_key"],
        how="left",
    )
    productor_exact_matches = int(dataset["precio_productor_provincia_usdkg"].notna().sum())

    dataset = dataset.merge(productor_nat_df, on="fecha", how="left")
    dataset = dataset.merge(
        agro_prov_df[
            [
                "fecha",
                "provincia_key",
                "fertilizantes_precio_promedio_provincia",
                "fertilizantes_precio_mediano_provincia",
                "fertilizantes_registros_provincia",
                "fertilizantes_ingredientes_activos_provincia",
            ]
        ],
        on=["fecha", "provincia_key"],
        how="left",
    )
    dataset = dataset.merge(agro_nat_df, on="fecha", how="left")
    dataset = dataset.merge(macro_df, on="fecha", how="left")
    dataset = dataset.merge(province_name_map, on=["provincia_key", "provincia"], how="left")

    dataset["producto"] = product_config["product_name"]
    dataset = dataset.sort_values(["provincia", "fecha"]).reset_index(drop=True)
    dataset = _add_calendar_features(dataset)
    dataset = _fill_exogenous_features(dataset)
    dataset = _add_group_lag_features(dataset)

    model_ready = dataset.dropna(subset=[TARGET_COLUMN] + RECOMMENDED_FEATURES).copy()
    model_ready = model_ready.sort_values(["fecha", "provincia"]).reset_index(drop=True)

    dataset_path = final_output_dir / "dataset_features.csv"
    metadata_path = final_output_dir / "metadata_features.json"

    model_ready.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    shared_provinces = province_name_map["provincia"].tolist()
    metadata = _build_metadata(
        product_config=product_config,
        base_df=dataset,
        final_df=model_ready,
        shared_provinces=shared_provinces,
        productor_exact_matches=productor_exact_matches,
        registros_iniciales=registros_iniciales,
    )
    write_json(metadata_path, metadata)

    return dataset_path, metadata_path


def build_preprocessing_summary_table(output_path: Path | None = None) -> Path:
    """Agrega el preprocessing_summary de los 3 productos en una sola tabla, para citar
    directamente en la tesis (punto C.9). Debe correrse despues de regenerar los 3
    datasets, ya que lee metadata_features.json de cada uno."""
    rows = []
    for product_id, product_config in PRODUCT_CONFIGS.items():
        metadata_path = product_config["output_dir"] / "metadata_features.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = metadata.get("preprocessing_summary", {})
        rows.append({"product_id": product_id, "producto": metadata.get("product_name"), **summary})

    table = pd.DataFrame(rows)
    destination = output_path or (Path(__file__).resolve().parents[1] / "outputs" / "feature_engineering_productos" / "preprocessing_summary.csv")
    ensure_directory(destination.parent)
    table.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination
