from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS


TARGET_COLUMN = "target_precio_mercado_usdkg"
PLOTS_DIR = MODEL_RESULTS_DIR / "plots"
PRODUCT_LABELS = {
    "maracuya": "Maracuyá",
    "papa_superchola": "Papa Superchola",
    "tomate_rinon_invernadero": "Tomate Riñón de Invernadero",
}
PRODUCT_COLORS = {
    "maracuya": "#d97706",
    "papa_superchola": "#2563eb",
    "tomate_rinon_invernadero": "#059669",
}


def load_monthly_product_prices() -> pd.DataFrame:
    frames = []
    for product_id, product in PRODUCTS.items():
        if not product.dataset_path.exists():
            raise FileNotFoundError(f"No existe el dataset: {product.dataset_path}")

        df = pd.read_csv(product.dataset_path, encoding="utf-8-sig")
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        monthly = (
            df.dropna(subset=["fecha", TARGET_COLUMN])
            .groupby("fecha", as_index=False)[TARGET_COLUMN]
            .mean()
            .sort_values("fecha")
        )
        monthly["product_id"] = product_id
        monthly["producto"] = PRODUCT_LABELS.get(product_id, product.display_name)
        frames.append(monthly)

    return pd.concat(frames, ignore_index=True)


def plot_historical_prices(output_path: Path) -> Path:
    data = load_monthly_product_prices()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for product_id in PRODUCTS:
        product_df = data.loc[data["product_id"] == product_id].sort_values("fecha")
        ax.plot(
            product_df["fecha"],
            product_df[TARGET_COLUMN],
            label=PRODUCT_LABELS.get(product_id, product_id),
            color=PRODUCT_COLORS.get(product_id),
            linewidth=2.4,
        )

    ax.set_title("Evolución mensual del precio promedio por producto", fontsize=15, weight="bold")
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Precio promedio de mercado (USD/kg)", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Producto", frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera la figura historica del precio promedio mensual por producto."
    )
    parser.add_argument(
        "--output",
        default=str(PLOTS_DIR / "figura_x_evolucion_precio_promedio_productos.png"),
        help="Ruta de salida del grafico PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_historical_prices(Path(args.output))
    print(f"Figura generada: {output_path}")


if __name__ == "__main__":
    main()
