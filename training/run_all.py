from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != TRAINING_DIR]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.arima.train import train_product as train_arima
from training.common.registry import rebuild_consolidated_results, selected_products
from training.lstm.train import train_product as train_lstm
from training.xgboost.train import train_product as train_xgboost


TRAINERS = {
    "xgboost": train_xgboost,
    "lstm": train_lstm,
    "arima": train_arima,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta entrenamientos por producto y modelo.")
    parser.add_argument("--product", default="all", help="Producto a entrenar o 'all'.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TRAINERS),
        choices=list(TRAINERS),
        help="Modelos a entrenar.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continua con otros entrenamientos si uno falla por dependencias o datos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for product in selected_products(args.product):
        for model_name in args.models:
            try:
                result = TRAINERS[model_name](product.product_id)
                rows.append({"model_name": model_name, **result, "status": "ok"})
            except Exception as exc:
                rows.append(
                    {
                        "product_id": product.product_id,
                        "model_name": model_name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if not args.continue_on_error:
                    raise

    rebuild_consolidated_results()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
