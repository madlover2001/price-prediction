from __future__ import annotations

import pandas as pd

from training.common.metrics import regression_metrics

# Corrige el punto C.3 de la revision del companero: XGBoost/LSTM ya compartian las
# mismas ventanas de validacion (fechas globales, agrupando todas las provincias), pero
# SARIMAX las recalculaba por provincia con cortes propios -- validation_rmse_mean no era
# comparable entre familias. Ahora los 3 modelos usan las mismas ventanas (ver
# training/arima/train.py, training/xgboost/train.py, training/lstm/train.py) y cada uno
# guarda sus predicciones out-of-fold (una por fila de cada ventana de validacion, sin
# solapes) en validation_predictions.csv. Este modulo intersecta esas predicciones por
# producto-provincia-fecha entre los 3 modelos y calcula un RMSE de validacion realmente
# comparable, como diagnostico secundario (la seleccion del "mejor modelo" usa el
# criterio operacional de C.6/C.7, ver training/common/registry.py).

KEY_COLUMNS = ["producto", "provincia", "fecha"]


def get_common_validation_predictions(product_id: str) -> dict[str, pd.DataFrame]:
    from training.common.registry import MODEL_NAMES, artifact_dir

    predictions_by_model = {}
    for model_name in MODEL_NAMES:
        path = artifact_dir(product_id, model_name, "full") / "validation_predictions.csv"
        if not path.exists():
            continue
        predictions = pd.read_csv(path, encoding="utf-8-sig")
        if predictions.empty:
            continue
        predictions["fecha"] = pd.to_datetime(predictions["fecha"])
        predictions_by_model[model_name] = predictions

    if len(predictions_by_model) < 2:
        return {}

    common_keys = None
    for predictions in predictions_by_model.values():
        keys = predictions[KEY_COLUMNS].drop_duplicates()
        common_keys = keys if common_keys is None else common_keys.merge(keys, on=KEY_COLUMNS, how="inner")
    if common_keys is None or common_keys.empty:
        return {}

    matched_by_model = {}
    for model_name, predictions in predictions_by_model.items():
        matched = (
            predictions.merge(common_keys, on=KEY_COLUMNS, how="inner")
            .sort_values(KEY_COLUMNS)
            .reset_index(drop=True)
        )
        if not matched.empty:
            matched_by_model[model_name] = matched
    return matched_by_model


def build_common_validation_window(product_id: str, product_name: str) -> tuple[pd.DataFrame, dict]:
    matched = get_common_validation_predictions(product_id)
    manifest = {"product_id": product_id, "models_available": list(matched), "common_validation_rows": 0}

    if len(matched) < 2:
        manifest["warning"] = "menos de 2 modelos con validation_predictions.csv disponibles"
        return pd.DataFrame(), manifest

    manifest["common_validation_rows"] = int(len(next(iter(matched.values()))))

    rows = []
    for model_name, predictions in matched.items():
        metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "model_name": model_name,
                "feature_set": "full",
                "validation_rmse_common": metrics["rmse"],
                "validation_mae_common": metrics["mae"],
                "validation_mape_common": metrics["mape"],
                "n": metrics["n_test"],
            }
        )

    return pd.DataFrame(rows), manifest


def run_common_validation_window_evaluation() -> None:
    from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS, write_json

    comparison_rows = []
    manifests = {}
    for product_id, product in PRODUCTS.items():
        comparison, manifest = build_common_validation_window(product_id, product.display_name)
        manifests[product_id] = manifest
        if not comparison.empty:
            comparison_rows.append(comparison)

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = MODEL_RESULTS_DIR / "common_validation_window.csv"
    manifest_path = MODEL_RESULTS_DIR / "common_validation_manifest.json"

    if comparison_rows:
        pd.concat(comparison_rows, ignore_index=True).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(comparison_path, index=False, encoding="utf-8-sig")
    write_json(manifest_path, manifests)
