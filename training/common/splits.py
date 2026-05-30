from __future__ import annotations

import pandas as pd


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
