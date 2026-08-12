from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from training.common.config import FEATURE_SETS, MODEL_NAMES

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT_DIR / "outputs" / "feature_engineering_productos"
MODELS_DIR = ROOT_DIR / "models"
MODEL_RESULTS_DIR = ROOT_DIR / "outputs" / "model_results"
PREDICTIONS_DIR = MODEL_RESULTS_DIR / "predictions"


@dataclass(frozen=True)
class ProductConfig:
    product_id: str
    display_name: str
    dataset_dir: Path

    @property
    def dataset_path(self) -> Path:
        return self.dataset_dir / "dataset_features.csv"

    @property
    def metadata_path(self) -> Path:
        return self.dataset_dir / "metadata_features.json"


PRODUCTS = {
    "papa_superchola": ProductConfig(
        product_id="papa_superchola",
        display_name="Papa Superchola",
        dataset_dir=DATASETS_DIR / "papa_superchola",
    ),
    "tomate_rinon_invernadero": ProductConfig(
        product_id="tomate_rinon_invernadero",
        display_name="Tomate Rinon de Invernadero",
        dataset_dir=DATASETS_DIR / "tomate_rinon_invernadero",
    ),
    "maracuya": ProductConfig(
        product_id="maracuya",
        display_name="Maracuya",
        dataset_dir=DATASETS_DIR / "maracuya",
    ),
}


def get_product(product_id: str) -> ProductConfig:
    if product_id not in PRODUCTS:
        valid = ", ".join(PRODUCTS)
        raise ValueError(f"Producto no configurado: {product_id}. Valores validos: {valid}")
    return PRODUCTS[product_id]


def selected_products(product_id: str | None) -> list[ProductConfig]:
    if not product_id or product_id == "all":
        return list(PRODUCTS.values())
    return [get_product(product_id)]


def artifact_dir(product_id: str, model_name: str, feature_set: str) -> Path:
    path = MODELS_DIR / product_id / model_name / feature_set
    path.mkdir(parents=True, exist_ok=True)
    return path


def prediction_path(product_id: str, model_name: str, feature_set: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return PREDICTIONS_DIR / f"{product_id}_{model_name}_{feature_set}_predictions.csv"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_result(
    product_id: str,
    model_name: str,
    feature_set: str,
    metrics: dict,
    metadata: dict,
    predictions: pd.DataFrame,
) -> None:
    directory = artifact_dir(product_id, model_name, feature_set)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "training_metadata.json", metadata)
    predictions.to_csv(prediction_path(product_id, model_name, feature_set), index=False, encoding="utf-8-sig")
    rebuild_consolidated_results()


def _metric_record(product_id: str, model_name: str, feature_set: str, metrics_path: Path) -> dict:
    metrics = read_json(metrics_path)
    return {
        "product_id": product_id,
        "product_name": PRODUCTS.get(product_id, ProductConfig(product_id, product_id, Path())).display_name,
        "model_name": model_name,
        "feature_set": feature_set,
        **metrics,
    }


def collect_metrics() -> pd.DataFrame:
    records: list[dict] = []
    for product_id in PRODUCTS:
        for model_name in MODEL_NAMES:
            for feature_set in FEATURE_SETS:
                metrics_path = MODELS_DIR / product_id / model_name / feature_set / "metrics.json"
                if metrics_path.exists():
                    records.append(_metric_record(product_id, model_name, feature_set, metrics_path))
    return pd.DataFrame(records)


def rebuild_consolidated_results() -> None:
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = collect_metrics()

    summary_path = MODEL_RESULTS_DIR / "metrics_summary.csv"
    comparison_path = MODEL_RESULTS_DIR / "comparison_by_product.csv"
    best_models_path = MODEL_RESULTS_DIR / "best_models.json"

    if metrics_df.empty:
        pd.DataFrame().to_csv(summary_path, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(comparison_path, index=False, encoding="utf-8-sig")
        write_json(best_models_path, {})
        return

    metrics_df = metrics_df.sort_values(
        ["product_id", "feature_set", "rmse", "mae", "model_name"]
    ).reset_index(drop=True)
    metrics_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # La seleccion del "mejor modelo" solo compite entre configuraciones "full" (la
    # ablacion "base" existe unicamente para medir el aporte de las exogenas, no para
    # ganar). El criterio es `operational_rmse_mean` -- desempeno agregado en pronostico
    # recursivo h=1..3 con carry-forward, evaluado dentro del 80% de desarrollo (nunca
    # toca el test) -- no el RMSE de un paso adelante. Corrige el punto C.6/C.7 de la
    # revision del companero: el prototipo promete horizontes de 1 a 3 meses, y el
    # "mejor modelo" retrospectivo de un paso no necesariamente es el mejor modelo
    # operacional. `validation_rmse_mean` (un paso, por familia) y `validation_rmse_common`
    # (ventana comun de validacion, ver common_validation_window.csv) se conservan como
    # campos secundarios de transparencia, no deciden. Ver docs/correcciones_docente.md.
    full_df = metrics_df[metrics_df["feature_set"] == "full"].copy()
    comparison_rows = []
    best_models = {}
    for product_id, group in full_df.groupby("product_id", sort=True):
        if "operational_rmse_mean" not in group.columns or group["operational_rmse_mean"].isna().all():
            ordered = group.sort_values(["rmse", "mae", "mape", "model_name"])
            selection_metric = "rmse"
        else:
            ordered = group.sort_values(["operational_rmse_mean", "rmse", "mae", "model_name"])
            selection_metric = "operational_rmse_mean"
        best = ordered.iloc[0]
        comparison_rows.append(best.to_dict())
        model_name = str(best["model_name"])
        best_models[product_id] = {
            "product_name": PRODUCTS[product_id].display_name,
            "best_model": model_name,
            "feature_set": "full",
            "selection_metric": selection_metric,
            "operational_rmse_mean": float(best.get("operational_rmse_mean", float("nan"))),
            "validation_rmse_mean": float(best.get("validation_rmse_mean", float("nan"))),
            "test_metrics": {
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "mape": float(best["mape"]),
                "r2": float(best["r2"]),
                "directional_accuracy": float(best["directional_accuracy"]),
            },
            "artifact_dir": str((MODELS_DIR / product_id / model_name / "full").relative_to(ROOT_DIR)),
            "prediction_file": str(prediction_path(product_id, model_name, "full").relative_to(ROOT_DIR)),
        }

    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    write_json(best_models_path, best_models)


def rebuild_ablation_summary() -> None:
    """Evidencia cuantitativa para el punto #1 del docente: cuanto mejora (o no) cada
    familia de modelo al pasar de `base` (solo historia del target) a `full` (+ exogenas)."""
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ablation_path = MODEL_RESULTS_DIR / "ablation_summary.csv"
    metrics_df = collect_metrics()

    if metrics_df.empty:
        pd.DataFrame().to_csv(ablation_path, index=False, encoding="utf-8-sig")
        return

    pivot = metrics_df.pivot_table(
        index=["product_id", "product_name", "model_name"],
        columns="feature_set",
        values=["rmse", "mae", "mape"],
    )
    pivot.columns = [f"{metric}_{feature_set}" for metric, feature_set in pivot.columns]
    pivot = pivot.reset_index()

    if "rmse_base" in pivot.columns and "rmse_full" in pivot.columns:
        pivot["delta_rmse_full_vs_base"] = pivot["rmse_base"] - pivot["rmse_full"]
        pivot["pct_improvement_full_vs_base"] = (
            pivot["delta_rmse_full_vs_base"] / pivot["rmse_base"] * 100
        )

    pivot.to_csv(ablation_path, index=False, encoding="utf-8-sig")
