from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


MONTH_MAP = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"No se pudo leer el archivo {path}")


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"No se pudo leer el archivo {path}")


def normalize_column_name(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def normalize_series_text(series: pd.Series, title_case: bool = False) -> pd.Series:
    cleaned = series.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    if title_case:
        cleaned = cleaned.str.title()
    return cleaned


def normalize_key(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [normalize_column_name(column) for column in renamed.columns]
    return renamed


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[column for column in df.columns if not column.startswith("unnamed")]].copy()


def add_monthly_date(df: pd.DataFrame, year_col: str = "ano", month_col: str = "mes") -> pd.DataFrame:
    dated = df.copy()
    dated[month_col] = normalize_series_text(dated[month_col]).str.lower()
    dated["mes_num"] = dated[month_col].map(MONTH_MAP)
    dated["fecha"] = pd.to_datetime(
        {
            "year": pd.to_numeric(dated[year_col], errors="coerce"),
            "month": pd.to_numeric(dated["mes_num"], errors="coerce"),
            "day": 1,
        },
        errors="coerce",
    )
    return dated


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace("$", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def most_common_value(series: pd.Series) -> str:
    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    if non_empty.empty:
        return ""
    return non_empty.mode().iloc[0]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
