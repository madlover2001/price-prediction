from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.data_loader import load_product_dataset
from training.common.metrics import regression_metrics
from training.common.registry import artifact_dir, selected_products, write_model_result
from training.common.splits import temporal_split_by_group, temporal_train_validation_split_by_group
from training.xgboost.config import (
    ENABLE_PARAM_TUNING,
    LAG_BLEND_FEATURE,
    LAG_BLEND_WEIGHT,
    MODEL_NAME,
    PARAM_GRID,
    PARAMS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


def select_best_params(train_df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[dict, list[dict]]:
    fit_df, validation_df = temporal_train_validation_split_by_group(
        train_df,
        train_ratio=VALIDATION_RATIO,
    )
    tuning_results = []

    for index, params in enumerate(PARAM_GRID):
        model = XGBRegressor(**params)
        model.fit(fit_df[feature_columns], fit_df[target_column])
        validation_pred = model.predict(validation_df[feature_columns])
        validation_metrics = regression_metrics(validation_df[target_column], validation_pred)
        tuning_results.append(
            {
                "candidate": index,
                "validation_rmse": validation_metrics["rmse"],
                "validation_mae": validation_metrics["mae"],
                "params": params,
            }
        )

    best = min(tuning_results, key=lambda row: (row["validation_rmse"], row["validation_mae"]))
    return best["params"], tuning_results


def blend_with_lag(raw_prediction, df: pd.DataFrame):
    return (1 - LAG_BLEND_WEIGHT) * raw_prediction + LAG_BLEND_WEIGHT * df[LAG_BLEND_FEATURE].to_numpy()


def train_product(product_id: str) -> dict:
    product = selected_products(product_id)[0]
    bundle = load_product_dataset(product)
    train_df, test_df = temporal_split_by_group(bundle.data, train_ratio=TRAIN_RATIO)
    if ENABLE_PARAM_TUNING:
        best_params, tuning_results = select_best_params(train_df, bundle.feature_columns, bundle.target_column)
    else:
        best_params, tuning_results = PARAMS, []

    model = XGBRegressor(**best_params)
    model.fit(train_df[bundle.feature_columns], train_df[bundle.target_column])

    raw_pred = model.predict(test_df[bundle.feature_columns])
    y_pred = blend_with_lag(raw_pred, test_df)
    metrics = regression_metrics(test_df[bundle.target_column], y_pred)

    predictions = test_df[["fecha", "producto", "provincia", bundle.target_column]].copy()
    predictions = predictions.rename(columns={bundle.target_column: "y_true"})
    predictions["y_pred_raw"] = raw_pred
    predictions["y_pred"] = y_pred
    predictions["model_name"] = MODEL_NAME
    predictions["product_id"] = product.product_id

    directory = artifact_dir(product.product_id, MODEL_NAME)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "product_id": product.product_id,
        "product_name": product.display_name,
        "model_name": MODEL_NAME,
        "model_file": str(model_path.relative_to(ROOT_DIR)),
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_ratio": TRAIN_RATIO,
        "params": best_params,
        "tuning_results": tuning_results,
        "validation_ratio_within_train": VALIDATION_RATIO,
        "enable_param_tuning": ENABLE_PARAM_TUNING,
        "lag_blend_feature": LAG_BLEND_FEATURE,
        "lag_blend_weight": LAG_BLEND_WEIGHT,
    }
    write_model_result(product.product_id, MODEL_NAME, metrics, metadata, predictions)
    return {"product_id": product.product_id, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena XGBoost por producto.")
    parser.add_argument("--product", default="all", help="Producto a entrenar o 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for product in selected_products(args.product):
        results.append(train_product(product.product_id))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
