from __future__ import annotations

import numpy as np
import pandas as pd

from training.common.config import (
    HOLDOUT_TEST_RATIO,
    MIN_VALIDATION_WINDOW_MONTHS,
    N_VALIDATION_WINDOWS,
)


def temporal_split_by_group(
    df: pd.DataFrame,
    group_col: str = "provincia",
    date_col: str = "fecha",
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, group in df.sort_values([group_col, date_col]).groupby(group_col, sort=False):
        split_index = int(len(group) * train_ratio)
        split_index = max(1, min(split_index, len(group) - 1))
        train_parts.append(group.iloc[:split_index])
        test_parts.append(group.iloc[split_index:])

    train_df = pd.concat(train_parts, ignore_index=True).sort_values([date_col, group_col]).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values([date_col, group_col]).reset_index(drop=True)
    validate_temporal_split(train_df, test_df, group_col, date_col)
    return train_df, test_df


def temporal_train_validation_split_by_group(
    df: pd.DataFrame,
    group_col: str = "provincia",
    date_col: str = "fecha",
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_parts = []
    validation_parts = []

    for _, group in df.sort_values([group_col, date_col]).groupby(group_col, sort=False):
        split_index = int(len(group) * train_ratio)
        split_index = max(1, min(split_index, len(group) - 1))
        fit_parts.append(group.iloc[:split_index])
        validation_parts.append(group.iloc[split_index:])

    fit_df = pd.concat(fit_parts, ignore_index=True).sort_values([date_col, group_col]).reset_index(drop=True)
    validation_df = (
        pd.concat(validation_parts, ignore_index=True)
        .sort_values([date_col, group_col])
        .reset_index(drop=True)
    )
    validate_temporal_split(fit_df, validation_df, group_col, date_col)
    return fit_df, validation_df


def validate_temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "provincia",
    date_col: str = "fecha",
) -> None:
    for group_name in sorted(set(train_df[group_col]).intersection(set(test_df[group_col]))):
        max_train_date = train_df.loc[train_df[group_col] == group_name, date_col].max()
        min_test_date = test_df.loc[test_df[group_col] == group_name, date_col].min()
        if min_test_date <= max_train_date:
            raise ValueError(
                f"Split temporal invalido para {group_name}: test empieza en {min_test_date}, "
                f"train termina en {max_train_date}"
            )


def compute_holdout_cutoffs(
    df: pd.DataFrame,
    group_col: str = "provincia",
    date_col: str = "fecha",
    test_ratio: float = HOLDOUT_TEST_RATIO,
) -> dict:
    """Fecha de corte (primera fecha de test) por grupo, calculada una sola vez.

    Los 3 modelos (xgboost, lstm, arima) llaman esta misma funcion antes de particionar,
    de modo que su holdout final de test cubra exactamente las mismas filas
    producto-provincia-fecha (ver docs/correcciones_docente.md, punto 3). LSTM difiere
    unicamente en las primeras WINDOW_SIZE-1 fechas de test de cada provincia, que no
    puede predecir sin historial previo: es una limitacion estructural documentada, no
    una diferencia de particion.
    """
    cutoffs = {}
    for group_name, group in df.sort_values([group_col, date_col]).groupby(group_col, sort=False):
        split_index = int(len(group) * (1 - test_ratio))
        split_index = max(1, min(split_index, len(group) - 1))
        cutoffs[group_name] = group.iloc[split_index][date_col]
    return cutoffs


def apply_holdout_cutoffs(
    df: pd.DataFrame,
    cutoffs: dict,
    group_col: str = "provincia",
    date_col: str = "fecha",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for group_name, group in df.sort_values([group_col, date_col]).groupby(group_col, sort=False):
        cutoff = cutoffs.get(group_name)
        if cutoff is None:
            continue
        train_parts.append(group[group[date_col] < cutoff])
        test_parts.append(group[group[date_col] >= cutoff])

    train_df = pd.concat(train_parts, ignore_index=True).sort_values([date_col, group_col]).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values([date_col, group_col]).reset_index(drop=True)
    validate_temporal_split(train_df, test_df, group_col, date_col)
    return train_df, test_df


def expanding_validation_windows(
    dev_df: pd.DataFrame,
    date_col: str = "fecha",
    n_windows: int = N_VALIDATION_WINDOWS,
    min_window_months: int = MIN_VALIDATION_WINDOW_MONTHS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Ventanas de validacion expansivas definidas por fecha a nivel producto.

    Cada ventana usa TODAS las provincias con datos hasta la fecha de corte para
    entrenar, y el siguiente tramo de fechas (todas las provincias) para validar. El
    numero de ventanas se reduce de 3 a 2 si el rango de fechas del 80% de desarrollo
    es demasiado corto para que cada bloque de validacion tenga al menos
    min_window_months meses (evita fragmentar productos con poco historial, como
    Papa Superchola). Nunca toca el 20% de holdout externo.
    """
    unique_dates = np.array(sorted(dev_df[date_col].unique()))

    windows = n_windows
    while windows >= 2:
        chunks = np.array_split(unique_dates, windows + 1)
        validation_chunks = chunks[1:]
        if windows == 2 or min(len(chunk) for chunk in validation_chunks) >= min_window_months:
            break
        windows -= 1

    chunks = np.array_split(unique_dates, windows + 1)

    splits = []
    for i in range(windows):
        train_cutoff = chunks[i][-1]
        val_dates = chunks[i + 1]
        if len(val_dates) == 0:
            continue
        train_window = dev_df[dev_df[date_col] <= train_cutoff]
        val_window = dev_df[dev_df[date_col].isin(val_dates)]
        if train_window.empty or val_window.empty:
            continue
        splits.append((train_window.reset_index(drop=True), val_window.reset_index(drop=True)))
    return splits
