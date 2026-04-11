from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILE = Path("outputs/precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv")
OUTPUT_DIR = Path("outputs/eda_productor")


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"No se pudo leer el archivo {path}")


def normalize_text(series: pd.Series, title_case: bool = False) -> pd.Series:
    normalized = series.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    if title_case:
        normalized = normalized.str.title()
    return normalized


def normalize_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    rename_map = {
        "año": "anio",
        "aÃ±o": "anio",
        "presentación": "presentacion",
        "presentaciÃ³n": "presentacion",
    }
    cleaned = cleaned.rename(columns=rename_map)

    required_columns = {"producto", "provincia"}
    missing_columns = required_columns.difference(cleaned.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Al archivo {source_name} le faltan columnas requeridas: {missing_str}")

    cleaned["producto"] = normalize_text(cleaned["producto"])
    cleaned["provincia"] = normalize_text(cleaned["provincia"], title_case=True)
    cleaned["fuente"] = source_name

    valid_rows = cleaned["producto"].ne("") & cleaned["provincia"].ne("")
    return cleaned.loc[valid_rows].copy()


def summarize_products(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("producto", as_index=False)
        .agg(
            registros=("producto", "size"),
            provincias_distintas=("provincia", "nunique"),
            provincias=("provincia", lambda values: " | ".join(sorted(set(values)))),
        )
        .sort_values(by=["registros", "provincias_distintas", "producto"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def summarize_provinces(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("provincia", as_index=False)
        .agg(
            registros=("provincia", "size"),
            productos_distintos=("producto", "nunique"),
        )
        .sort_values(by=["registros", "productos_distintos", "provincia"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_analysis(file_path: Path, output_dir: Path) -> None:
    raw_df = read_csv_with_fallback(file_path)
    normalized_df = normalize_dataframe(raw_df, file_path.name)

    product_summary = summarize_products(normalized_df)
    province_summary = summarize_provinces(normalized_df)

    save_summary(product_summary, output_dir / f"{file_path.stem}_resumen_productos.csv")
    save_summary(province_summary, output_dir / f"{file_path.stem}_resumen_provincias.csv")

    print(f"\nArchivo analizado: {file_path}")
    print(f"Registros validos: {len(normalized_df):,}")
    print(f"Productos distintos: {normalized_df['producto'].nunique():,}")
    print(f"Provincias distintas: {normalized_df['provincia'].nunique():,}")
    print("\nTop 15 productos por numero de registros:")
    print(product_summary.head(15).to_string(index=False))
    print(f"\nArchivos exportados en: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA basico para contar productos y provincias en precios productor ponderado."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Archivo CSV a analizar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directorio donde se guardaran los resumenes en CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.file, args.output_dir)
