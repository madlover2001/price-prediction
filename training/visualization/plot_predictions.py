from __future__ import annotations

import argparse
import sys
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS


PREDICTIONS_DIR = MODEL_RESULTS_DIR / "predictions"
PLOTS_DIR = MODEL_RESULTS_DIR / "plots"
MODEL_ORDER = ["xgboost", "arima", "lstm"]
MODEL_COLORS = {
    "real": "#111827",
    "xgboost": "#2563eb",
    "arima": "#dc2626",
    "lstm": "#059669",
}


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def load_predictions(product_id: str) -> pd.DataFrame:
    frames = []
    for model_name in MODEL_ORDER:
        path = PREDICTIONS_DIR / f"{product_id}_{model_name}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["model_name"] = model_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No hay predicciones para {product_id} en {PREDICTIONS_DIR}")

    return pd.concat(frames, ignore_index=True).dropna(subset=["fecha"])


def plot_product_province(product_id: str, province: str, output_dir: Path) -> Path:
    df = load_predictions(product_id)
    province_df = df.loc[df["provincia"].str.lower() == province.lower()].copy()
    if province_df.empty:
        available = ", ".join(sorted(df["provincia"].dropna().unique()))
        raise ValueError(f"No hay predicciones para provincia '{province}'. Disponibles: {available}")

    province_name = province_df["provincia"].mode().iloc[0]
    product_name = province_df["producto"].mode().iloc[0]
    real_df = (
        province_df[["fecha", "y_true"]]
        .drop_duplicates("fecha")
        .sort_values("fecha")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        real_df["fecha"],
        real_df["y_true"],
        label="Real",
        color=MODEL_COLORS["real"],
        linewidth=2.5,
    )

    for model_name in MODEL_ORDER:
        model_df = province_df.loc[province_df["model_name"] == model_name].sort_values("fecha")
        if model_df.empty:
            continue
        ax.plot(
            model_df["fecha"],
            model_df["y_pred"],
            label=model_name.upper(),
            color=MODEL_COLORS[model_name],
            linewidth=1.8,
            alpha=0.9,
        )

    ax.set_title(f"{product_name} - {province_name}: precio real vs predicciones")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio USD/kg")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    product_dir = output_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)
    output_path = product_dir / f"{product_id}_{slugify(province_name)}_comparacion.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_product_summary(product_id: str, output_dir: Path) -> Path:
    df = load_predictions(product_id)
    product_name = df["producto"].mode().iloc[0]
    summary = (
        df.groupby(["fecha", "model_name"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        .sort_values("fecha")
    )
    real_df = summary[["fecha", "y_true"]].drop_duplicates("fecha").sort_values("fecha")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        real_df["fecha"],
        real_df["y_true"],
        label="Real promedio provincias",
        color=MODEL_COLORS["real"],
        linewidth=2.5,
    )

    for model_name in MODEL_ORDER:
        model_df = summary.loc[summary["model_name"] == model_name].sort_values("fecha")
        if model_df.empty:
            continue
        ax.plot(
            model_df["fecha"],
            model_df["y_pred"],
            label=model_name.upper(),
            color=MODEL_COLORS[model_name],
            linewidth=1.8,
            alpha=0.9,
        )

    ax.set_title(f"{product_name}: promedio provincial real vs predicciones")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio USD/kg")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    product_dir = output_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)
    output_path = product_dir / f"{product_id}_resumen_promedio_provincias.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_plots(product_id: str | None, province: str | None, output_dir: Path) -> list[Path]:
    product_ids = [product_id] if product_id else list(PRODUCTS)
    output_paths = []

    for current_product in product_ids:
        df = load_predictions(current_product)
        output_paths.append(plot_product_summary(current_product, output_dir))

        provinces = [province] if province else sorted(df["provincia"].dropna().unique())
        for current_province in provinces:
            output_paths.append(plot_product_province(current_product, current_province, output_dir))

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficas comparando precio real vs predicciones ARIMA, LSTM y XGBoost."
    )
    parser.add_argument("--product", choices=list(PRODUCTS), help="Producto especifico. Si se omite, grafica todos.")
    parser.add_argument("--province", help="Provincia especifica. Si se omite, grafica todas.")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_plots(args.product, args.province, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
