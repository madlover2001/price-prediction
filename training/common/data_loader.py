from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from training.common.registry import ProductConfig, read_json


@dataclass
class ProductDataset:
    config: ProductConfig
    data: pd.DataFrame
    metadata: dict
    feature_columns: list[str]
    target_column: str
    date_column: str
    feature_set: str = "full"


REQUIRED_COLUMNS = {"fecha", "provincia", "provincia_id", "target_precio_mercado_usdkg"}


def load_product_dataset(config: ProductConfig, feature_set: str = "full") -> ProductDataset:
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset: {config.dataset_path}")
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata: {config.metadata_path}")
    if feature_set not in ("base", "full"):
        raise ValueError(f"feature_set invalido: {feature_set}. Valores validos: base, full")

    metadata = read_json(config.metadata_path)
    df = pd.read_csv(config.dataset_path, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values(["provincia", "fecha"]).reset_index(drop=True)

    target_column = metadata.get("target_column", "target_precio_mercado_usdkg")
    date_column = metadata.get("date_column", "fecha")
    # base = solo historia propia del target + calendario (para medir el aporte real de
    # las exogenas via ablacion, ver docs/correcciones_docente.md punto 1). full = dataset
    # completo, es la configuracion que compite como mejor modelo por producto. Si el
    # dataset no fue regenerado con las listas nuevas, cae a recommended_model_features.
    feature_key = f"{feature_set}_model_features"
    feature_columns = list(metadata.get(feature_key, metadata.get("recommended_model_features", [])))

    validate_dataset(df, feature_columns, target_column, date_column)

    return ProductDataset(
        config=config,
        data=df,
        metadata=metadata,
        feature_columns=feature_columns,
        target_column=target_column,
        date_column=date_column,
        feature_set=feature_set,
    )


def validate_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    date_column: str,
) -> None:
    missing_required = REQUIRED_COLUMNS.difference(df.columns)
    if missing_required:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_required)}")

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Faltan features recomendadas: {missing_features}")

    missing_core = [column for column in [target_column, date_column] if column not in df.columns]
    if missing_core:
        raise ValueError(f"Faltan columnas centrales: {missing_core}")

    if df[date_column].isna().any():
        raise ValueError("La columna fecha contiene valores no parseables.")

    numeric_columns = feature_columns + [target_column]
    non_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Columnas esperadas como numericas no lo son: {non_numeric}")

    missing_values = df[numeric_columns].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(f"Hay valores faltantes en variables de entrenamiento: {missing_values.to_dict()}")
