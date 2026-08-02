from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.config import FEATURE_SETS, HOLDOUT_TEST_RATIO
from training.common.data_loader import load_product_dataset
from training.common.metrics import regression_metrics
from training.common.registry import artifact_dir, selected_products, write_model_result
from training.common.splits import apply_holdout_cutoffs, compute_holdout_cutoffs, expanding_validation_windows
from training.xgboost.config import (
    ENABLE_PARAM_TUNING,
    LAG_BLEND_FEATURE,
    LAG_BLEND_WEIGHT,
    MODEL_NAME,
    PARAM_GRID,
    PARAMS,
)


def _validation_rmse(params: dict, windows: list, feature_columns: list[str], target_column: str) -> tuple[float, float]:
    window_rmses, window_maes = [], []
    for fit_df, validation_df in windows:
        model = XGBRegressor(**params)
        model.fit(fit_df[feature_columns], fit_df[target_column])
        validation_pred = model.predict(validation_df[feature_columns])
        window_metrics = regression_metrics(validation_df[target_column], validation_pred)
        window_rmses.append(window_metrics["rmse"])
        window_maes.append(window_metrics["mae"])
    if not window_rmses:
        return math.nan, math.nan
    return float(np.mean(window_rmses)), float(np.mean(window_maes))


def select_best_params(
    dev_df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[dict, float, list[dict]]:
    windows = expanding_validation_windows(dev_df)
    tuning_results = []

    for index, params in enumerate(PARAM_GRID):
        mean_rmse, mean_mae = _validation_rmse(params, windows, feature_columns, target_column)
        tuning_results.append(
            {
                "candidate": index,
                "validation_rmse_mean": mean_rmse,
                "validation_mae_mean": mean_mae,
                "n_windows": len(windows),
                "params": params,
            }
        )

    valid_results = [row for row in tuning_results if not math.isnan(row["validation_rmse_mean"])]
    if not valid_results:
        return PARAMS, math.nan, tuning_results

    best = min(valid_results, key=lambda row: (row["validation_rmse_mean"], row["validation_mae_mean"]))
    return best["params"], best["validation_rmse_mean"], tuning_results


def blend_with_lag(raw_prediction, df: pd.DataFrame):
    return (1 - LAG_BLEND_WEIGHT) * raw_prediction + LAG_BLEND_WEIGHT * df[LAG_BLEND_FEATURE].to_numpy()


def train_product(product_id: str, feature_set: str) -> dict:
    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product, feature_set=feature_set)

    cutoffs = compute_holdout_cutoffs(bundle.data, test_ratio=HOLDOUT_TEST_RATIO)
    train_df, test_df = apply_holdout_cutoffs(bundle.data, cutoffs)

    if ENABLE_PARAM_TUNING:
        best_params, validation_rmse_mean, tuning_results = select_best_params(
            train_df, bundle.feature_columns, bundle.target_column
        )
    else:
        windows = expanding_validation_windows(train_df)
        validation_rmse_mean, _ = _validation_rmse(PARAMS, windows, bundle.feature_columns, bundle.target_column)
        best_params, tuning_results = PARAMS, []

    model = XGBRegressor(**best_params)
    model.fit(train_df[bundle.feature_columns], train_df[bundle.target_column])

    raw_pred = model.predict(test_df[bundle.feature_columns])
    y_pred = blend_with_lag(raw_pred, test_df)
    metrics = regression_metrics(test_df[bundle.target_column], y_pred)
    metrics["validation_rmse_mean"] = validation_rmse_mean

    predictions = test_df[["fecha", "producto", "provincia", bundle.target_column]].copy()
    predictions = predictions.rename(columns={bundle.target_column: "y_true"})
    predictions["y_pred_raw"] = raw_pred
    predictions["y_pred"] = y_pred
    predictions["model_name"] = MODEL_NAME
    predictions["product_id"] = product.product_id
    predictions["feature_set"] = feature_set

    directory = artifact_dir(product.product_id, MODEL_NAME, feature_set)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "product_id": product.product_id,
        "product_name": product.display_name,
        "model_name": MODEL_NAME,
        "feature_set": feature_set,
        "model_file": str(model_path.relative_to(ROOT_DIR)),
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "holdout_test_ratio": HOLDOUT_TEST_RATIO,
        "params": best_params,
        "tuning_results": tuning_results,
        "validation_rmse_mean": validation_rmse_mean,
        "n_validation_windows": len(expanding_validation_windows(train_df)),
        "enable_param_tuning": ENABLE_PARAM_TUNING,
        "lag_blend_feature": LAG_BLEND_FEATURE,
        "lag_blend_weight": LAG_BLEND_WEIGHT,
    }
    write_model_result(product.product_id, MODEL_NAME, feature_set, metrics, metadata, predictions)
    return {"product_id": product.product_id, "feature_set": feature_set, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena XGBoost por producto.")
    parser.add_argument("--product", default="all", help="Producto a entrenar o 'all'.")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for product in selected_products(args.product):
        for feature_set in args.feature_sets:
            results.append(train_product(product.product_id, feature_set))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
