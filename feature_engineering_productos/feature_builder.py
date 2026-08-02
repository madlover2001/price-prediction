from __future__ import annotations

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

EXOGENOUS_CONTEXT_FEATURES = [
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
FULL_MODEL_FEATURES = BASE_MODEL_FEATURES + EXOGENOUS_CONTEXT_FEATURES + EXOGENOUS_LAG_FEATURES

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


def _build_metadata(
    product_config: dict,
    base_df: pd.DataFrame,
    final_df: pd.DataFrame,
    shared_provinces: list[str],
    productor_exact_matches: int,
) -> dict:
    missing_counts = final_df.isna().sum()
    missing_counts = {key: int(value) for key, value in missing_counts.items() if int(value) > 0}

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
            "market_context": ["mercados_observaciones", "mercados_distintos", "tipos_mercado_distintos"],
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
    )
    write_json(metadata_path, metadata)

    return dataset_path, metadata_path
