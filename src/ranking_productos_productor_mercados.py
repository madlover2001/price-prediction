from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_PRODUCTOR_FILE = Path("outputs/precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv")
DEFAULT_MERCADOS_FILE = Path(
    "outputs/precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_Mercados12-25.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/ranking_productor_mercados")


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


def normalize_key(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def most_common_value(series: pd.Series) -> str:
    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    if non_empty.empty:
        return ""
    return non_empty.mode().iloc[0]


def prepare_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    rename_map = {
        "año": "anio",
        "aÃ±o": "anio",
        "cantón": "canton",
        "cantÃ³n": "canton",
        "presentación": "presentacion",
        "presentaciÃ³n": "presentacion",
    }
    cleaned = cleaned.rename(columns=rename_map)

    unnamed_columns = [column for column in cleaned.columns if column.lower().startswith("unnamed")]
    if unnamed_columns:
        cleaned = cleaned.drop(columns=unnamed_columns)

    required_columns = {"producto", "provincia"}
    missing_columns = required_columns.difference(cleaned.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Al archivo {source_label} le faltan columnas requeridas: {missing_str}")

    cleaned["producto"] = normalize_text(cleaned["producto"])
    cleaned["provincia"] = normalize_text(cleaned["provincia"], title_case=True)
    cleaned = cleaned.loc[cleaned["producto"].ne("") & cleaned["provincia"].ne("")].copy()

    cleaned["producto_key"] = cleaned["producto"].map(normalize_key)
    cleaned["provincia_key"] = cleaned["provincia"].map(normalize_key)
    cleaned["fuente"] = source_label
    return cleaned


def build_product_province_counts(df: pd.DataFrame, source_prefix: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["producto_key", "provincia_key"], as_index=False)
        .agg(
            producto=("producto", most_common_value),
            provincia=("provincia", most_common_value),
            registros=("producto", "size"),
        )
        .rename(
            columns={
                "producto": f"producto_{source_prefix}",
                "provincia": f"provincia_{source_prefix}",
                "registros": f"registros_{source_prefix}",
            }
        )
    )
    return grouped


def rank_shared_products(
    productor_counts: pd.DataFrame,
    mercados_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = productor_counts.merge(
        mercados_counts,
        on=["producto_key", "provincia_key"],
        how="outer",
    )

    merged["registros_productor"] = merged["registros_productor"].fillna(0).astype(int)
    merged["registros_mercados"] = merged["registros_mercados"].fillna(0).astype(int)

    product_summary = (
        merged.groupby("producto_key", as_index=False)
        .agg(
            producto_productor=("producto_productor", most_common_value),
            producto_mercados=("producto_mercados", most_common_value),
            registros_productor=("registros_productor", "sum"),
            registros_mercados=("registros_mercados", "sum"),
            provincias_productor=("registros_productor", lambda s: int((s > 0).sum())),
            provincias_mercados=("registros_mercados", lambda s: int((s > 0).sum())),
            provincias_en_ambas=("provincia_key", lambda s: 0),
            provincias_union=("provincia_key", "nunique"),
        )
    )

    shared_province_counts = (
        merged.loc[(merged["registros_productor"] > 0) & (merged["registros_mercados"] > 0)]
        .groupby("producto_key")
        .size()
        .rename("provincias_en_ambas")
        .reset_index()
    )
    product_summary = product_summary.drop(columns=["provincias_en_ambas"]).merge(
        shared_province_counts,
        on="producto_key",
        how="left",
    )
    product_summary["provincias_en_ambas"] = product_summary["provincias_en_ambas"].fillna(0).astype(int)

    product_summary = product_summary.loc[
        (product_summary["registros_productor"] > 0) & (product_summary["registros_mercados"] > 0)
    ].copy()

    product_summary["producto_recomendado"] = product_summary["producto_productor"].where(
        product_summary["producto_productor"].ne(""),
        product_summary["producto_mercados"],
    )
    product_summary["registros_total"] = (
        product_summary["registros_productor"] + product_summary["registros_mercados"]
    )
    product_summary["registros_balance"] = product_summary[
        ["registros_productor", "registros_mercados"]
    ].min(axis=1)
    product_summary["score_modelado"] = (
        product_summary["registros_total"]
        + 2 * product_summary["registros_balance"]
        + 25 * product_summary["provincias_en_ambas"]
    )

    product_summary = product_summary.sort_values(
        by=["score_modelado", "registros_balance", "registros_total", "producto_recomendado"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return product_summary, merged


def rank_influential_provinces(shared_product_province: pd.DataFrame, product_ranking: pd.DataFrame) -> pd.DataFrame:
    shared_keys = set(product_ranking["producto_key"])
    filtered = shared_product_province.loc[shared_product_province["producto_key"].isin(shared_keys)].copy()

    filtered["producto_recomendado"] = filtered["producto_productor"].where(
        filtered["producto_productor"].fillna("").ne(""),
        filtered["producto_mercados"],
    )
    filtered["provincia_recomendada"] = filtered["provincia_productor"].where(
        filtered["provincia_productor"].fillna("").ne(""),
        filtered["provincia_mercados"],
    )
    filtered["registros_total"] = filtered["registros_productor"] + filtered["registros_mercados"]
    filtered["presente_en_ambas_fuentes"] = (
        (filtered["registros_productor"] > 0) & (filtered["registros_mercados"] > 0)
    ).astype(int)

    province_summary = (
        filtered.groupby("provincia_key", as_index=False)
        .agg(
            provincia=("provincia_recomendada", most_common_value),
            registros_productor=("registros_productor", "sum"),
            registros_mercados=("registros_mercados", "sum"),
            registros_total=("registros_total", "sum"),
            productos_distintos=("producto_key", "nunique"),
            productos_en_ambas=("presente_en_ambas_fuentes", "sum"),
        )
    )
    province_summary["registros_balance"] = province_summary[
        ["registros_productor", "registros_mercados"]
    ].min(axis=1)
    province_summary["score_influencia"] = (
        province_summary["registros_total"]
        + 2 * province_summary["registros_balance"]
        + 25 * province_summary["productos_en_ambas"]
    )

    province_top_products = (
        filtered.groupby(["provincia_key", "producto_recomendado"], as_index=False)
        .agg(registros_total=("registros_total", "sum"))
        .sort_values(["provincia_key", "registros_total", "producto_recomendado"], ascending=[True, False, True])
    )
    top_products_by_province = (
        province_top_products.groupby("provincia_key")["producto_recomendado"]
        .apply(lambda s: " | ".join(s.head(5)))
        .rename("top_5_productos")
        .reset_index()
    )

    province_summary = province_summary.merge(top_products_by_province, on="provincia_key", how="left")
    province_summary = province_summary.sort_values(
        by=["score_influencia", "productos_en_ambas", "registros_total", "provincia"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return province_summary


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_analysis(productor_file: Path, mercados_file: Path, output_dir: Path) -> None:
    productor_df = prepare_dataframe(read_csv_with_fallback(productor_file), "productor")
    mercados_df = prepare_dataframe(read_csv_with_fallback(mercados_file), "mercados")

    productor_counts = build_product_province_counts(productor_df, "productor")
    mercados_counts = build_product_province_counts(mercados_df, "mercados")

    product_ranking, shared_product_province = rank_shared_products(productor_counts, mercados_counts)
    province_ranking = rank_influential_provinces(shared_product_province, product_ranking)

    save_csv(product_ranking, output_dir / "ranking_productos_compartidos.csv")
    save_csv(province_ranking, output_dir / "ranking_provincias_influyentes.csv")

    print("\nTOP 10 PRODUCTOS COMPARTIDOS PARA MODELADO")
    print(product_ranking[
        [
            "producto_recomendado",
            "registros_productor",
            "registros_mercados",
            "registros_total",
            "registros_balance",
            "provincias_en_ambas",
            "score_modelado",
        ]
    ].head(10).to_string(index=False))

    print("\nTOP 5 PROVINCIAS MAS INFLUYENTES")
    print(province_ranking[
        [
            "provincia",
            "registros_productor",
            "registros_mercados",
            "registros_total",
            "productos_distintos",
            "productos_en_ambas",
            "score_influencia",
            "top_5_productos",
        ]
    ].head(5).to_string(index=False))

    print(f"\nArchivos exportados en: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ranking de productos compartidos entre productor y mercados, y provincias mas influyentes."
    )
    parser.add_argument(
        "--productor-file",
        type=Path,
        default=DEFAULT_PRODUCTOR_FILE,
        help="CSV de precios productor ponderado.",
    )
    parser.add_argument(
        "--mercados-file",
        type=Path,
        default=DEFAULT_MERCADOS_FILE,
        help="CSV de precios de mercados mayoristas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde se guardaran los rankings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.productor_file, args.mercados_file, args.output_dir)
