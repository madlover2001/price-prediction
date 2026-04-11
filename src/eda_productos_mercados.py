from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILES = [
    Path("outputs/precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_Mercados12-25.csv"),
    Path("outputs/precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_bodegas_12-25.csv"),
]

OUTPUT_DIR = Path("outputs/eda_productos")


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
        "cantón": "canton",
        "cantÃ³n": "canton",
        "uni_med": "unidad_medida",
        "precio_promedio_usd": "promedio_de_precio_usd",
    }
    cleaned = cleaned.rename(columns=rename_map)

    unnamed_columns = [column for column in cleaned.columns if column.lower().startswith("unnamed")]
    if unnamed_columns:
        cleaned = cleaned.drop(columns=unnamed_columns)

    required_columns = {"producto", "provincia"}
    missing_columns = required_columns.difference(cleaned.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Al archivo {source_name} le faltan columnas requeridas: {missing_str}")

    cleaned["producto"] = normalize_text(cleaned["producto"])
    cleaned["provincia"] = normalize_text(cleaned["provincia"], title_case=True)
    cleaned["fuente"] = source_name

    valid_rows = cleaned["producto"].ne("") & cleaned["provincia"].ne("")
    cleaned = cleaned.loc[valid_rows].copy()

    return cleaned


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


def run_analysis(files: list[Path], output_dir: Path) -> None:
    normalized_frames: list[pd.DataFrame] = []

    for file_path in files:
        raw_df = read_csv_with_fallback(file_path)
        normalized_df = normalize_dataframe(raw_df, file_path.name)
        normalized_frames.append(normalized_df)

        product_summary = summarize_products(normalized_df)
        province_summary = summarize_provinces(normalized_df)

        save_summary(product_summary, output_dir / f"{file_path.stem}_resumen_productos.csv")
        save_summary(province_summary, output_dir / f"{file_path.stem}_resumen_provincias.csv")

        print(f"\nArchivo analizado: {file_path}")
        print(f"Registros validos: {len(normalized_df):,}")
        print(f"Productos distintos: {normalized_df['producto'].nunique():,}")
        print(f"Provincias distintas: {normalized_df['provincia'].nunique():,}")
        print("\nTop 10 productos por numero de registros:")
        print(product_summary.head(10).to_string(index=False))

    consolidated_df = pd.concat(normalized_frames, ignore_index=True)
    consolidated_products = summarize_products(consolidated_df)
    consolidated_provinces = summarize_provinces(consolidated_df)

    save_summary(consolidated_products, output_dir / "consolidado_resumen_productos.csv")
    save_summary(consolidated_provinces, output_dir / "consolidado_resumen_provincias.csv")

    print("\n" + "=" * 80)
    print("RESUMEN CONSOLIDADO")
    print("=" * 80)
    print(f"Registros validos: {len(consolidated_df):,}")
    print(f"Productos distintos: {consolidated_df['producto'].nunique():,}")
    print(f"Provincias distintas: {consolidated_df['provincia'].nunique():,}")
    print("\nTop 15 productos por numero de registros:")
    print(consolidated_products.head(15).to_string(index=False))
    print(f"\nArchivos exportados en: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA basico para contar productos y provincias en mercados mayoristas y bodegas."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=DEFAULT_FILES,
        help="Lista de archivos CSV a analizar.",
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
    run_analysis(args.files, args.output_dir)
