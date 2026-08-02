from __future__ import annotations

import joblib
import pandas as pd

from training.common.config import FEATURE_SETS

# Responde a "las afirmaciones... exceden la evidencia disponible al no incluir
# interpretabilidad" (ver docs/correcciones_docente.md). Se extrae la importancia de
# variables (gain-based, nativa de XGBoost) de los modelos ya entrenados -- no requiere
# reentrenar -- como evidencia minima de que variables impulsan cada prediccion.


def run_xgboost_feature_importance() -> None:
    from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS, artifact_dir, read_json

    rows = []
    for product_id, product in PRODUCTS.items():
        for feature_set in FEATURE_SETS:
            directory = artifact_dir(product_id, "xgboost", feature_set)
            model_path = directory / "model.joblib"
            metadata_path = directory / "training_metadata.json"
            if not model_path.exists() or not metadata_path.exists():
                continue

            model = joblib.load(model_path)
            metadata = read_json(metadata_path)
            feature_columns = metadata["feature_columns"]
            importances = model.feature_importances_

            for feature, importance in zip(feature_columns, importances):
                rows.append(
                    {
                        "product_id": product_id,
                        "product_name": product.display_name,
                        "feature_set": feature_set,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["importance_rank"] = df.groupby(["product_id", "feature_set"])["importance"].rank(
            ascending=False, method="first"
        )
        df = df.sort_values(["product_id", "feature_set", "importance_rank"]).reset_index(drop=True)

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODEL_RESULTS_DIR / "xgboost_feature_importance.csv", index=False, encoding="utf-8-sig")
