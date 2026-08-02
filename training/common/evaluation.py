from __future__ import annotations

import pandas as pd

from training.common.metrics import regression_metrics

# Corrige el punto #3 del docente ("los modelos no se evaluan sobre exactamente los
# mismos datos"): en vez de forzar splits identicos entre modelos -- imposible sin
# romper la naturaleza de LSTM (necesita WINDOW_SIZE de historial) y de SARIMAX por
# provincia (exige MIN_OBSERVATIONS) -- se intersectan las predicciones de test por
# producto-provincia-fecha entre los 3 modelos y las metricas primarias se calculan
# sobre esa interseccion comun. Solo aplica a la configuracion "full" (la que compite
# como mejor modelo); las metricas por modelo sobre su test completo se conservan en
# metrics_summary.csv como tabla secundaria de transparencia.

KEY_COLUMNS = ["producto", "provincia", "fecha"]


def common_evaluation_window(predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_keys = None
    for predictions in predictions_by_model.values():
        keys = predictions[KEY_COLUMNS].drop_duplicates()
        common_keys = keys if common_keys is None else common_keys.merge(keys, on=KEY_COLUMNS, how="inner")
    if common_keys is None:
        return pd.DataFrame(columns=KEY_COLUMNS)
    return common_keys.reset_index(drop=True)


def get_common_window_predictions(product_id: str) -> dict[str, pd.DataFrame]:
    """Predicciones `full` de cada modelo, recortadas a la interseccion comun de
    producto-provincia-fecha. Filas alineadas por indice entero (mismo orden en todos
    los modelos), lo que permite comparar y bootstrapear par a par."""
    from training.common.registry import MODEL_NAMES, prediction_path

    predictions_by_model = {}
    for model_name in MODEL_NAMES:
        path = prediction_path(product_id, model_name, "full")
        if not path.exists():
            continue
        predictions = pd.read_csv(path, encoding="utf-8-sig")
        predictions["fecha"] = pd.to_datetime(predictions["fecha"])
        predictions_by_model[model_name] = predictions

    if len(predictions_by_model) < 2:
        return {}

    common_keys = common_evaluation_window(predictions_by_model)
    if common_keys.empty:
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


def build_common_window_comparison(product_id: str, product_name: str) -> tuple[pd.DataFrame, dict]:
    from training.common.registry import MODEL_NAMES, prediction_path

    predictions_by_model = {}
    for model_name in MODEL_NAMES:
        path = prediction_path(product_id, model_name, "full")
        if not path.exists():
            continue
        predictions = pd.read_csv(path, encoding="utf-8-sig")
        predictions["fecha"] = pd.to_datetime(predictions["fecha"])
        predictions_by_model[model_name] = predictions

    manifest = {
        "product_id": product_id,
        "models_available": list(predictions_by_model),
        "common_test_rows": 0,
        "dropped_rows_by_model": {},
    }

    if len(predictions_by_model) < 2:
        manifest["warning"] = "menos de 2 modelos con predicciones full disponibles"
        return pd.DataFrame(), manifest

    common_keys = common_evaluation_window(predictions_by_model)
    manifest["common_test_rows"] = int(len(common_keys))

    rows = []
    for model_name, predictions in predictions_by_model.items():
        matched = predictions.merge(common_keys, on=KEY_COLUMNS, how="inner")
        manifest["dropped_rows_by_model"][model_name] = int(len(predictions) - len(matched))
        if matched.empty:
            continue
        metrics = regression_metrics(matched["y_true"], matched["y_pred"])
        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "model_name": model_name,
                "feature_set": "full",
                **metrics,
            }
        )

    return pd.DataFrame(rows), manifest


def run_common_window_evaluation() -> None:
    from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS, write_json

    comparison_rows = []
    manifests = {}
    for product_id, product in PRODUCTS.items():
        comparison, manifest = build_common_window_comparison(product_id, product.display_name)
        manifests[product_id] = manifest
        if not comparison.empty:
            comparison_rows.append(comparison)

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = MODEL_RESULTS_DIR / "common_window_comparison.csv"
    manifest_path = MODEL_RESULTS_DIR / "common_window_manifest.json"

    if comparison_rows:
        pd.concat(comparison_rows, ignore_index=True).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(comparison_path, index=False, encoding="utf-8-sig")
    write_json(manifest_path, manifests)
