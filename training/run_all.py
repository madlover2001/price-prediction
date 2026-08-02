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
from training.common.config import FEATURE_SETS
from training.common.evaluation import run_common_window_evaluation
from training.common.horizon_runner import run_horizon_evaluation
from training.common.interpretability import run_xgboost_feature_importance
from training.common.registry import rebuild_ablation_summary, rebuild_consolidated_results, selected_products
from training.common.uncertainty import run_rmse_uncertainty
from training.lstm.train import train_product as train_lstm
from training.xgboost.train import train_product as train_xgboost


TRAINERS = {
    "xgboost": train_xgboost,
    "lstm": train_lstm,
    "arima": train_arima,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta entrenamientos por producto, modelo y feature set.")
    parser.add_argument("--product", default="all", help="Producto a entrenar o 'all'.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TRAINERS),
        choices=list(TRAINERS),
        help="Modelos a entrenar.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
        help="Configuraciones de features a entrenar (ablacion base vs full).",
    )
    parser.add_argument(
        "--skip-horizons",
        action="store_true",
        help="Omite la evaluacion de horizontes 1-3 meses (mas rapido para pruebas).",
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
            for feature_set in args.feature_sets:
                try:
                    result = TRAINERS[model_name](product.product_id, feature_set)
                    rows.append({"model_name": model_name, **result, "status": "ok"})
                except Exception as exc:
                    rows.append(
                        {
                            "product_id": product.product_id,
                            "model_name": model_name,
                            "feature_set": feature_set,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    if not args.continue_on_error:
                        raise

    rebuild_consolidated_results()
    rebuild_ablation_summary()
    run_common_window_evaluation()
    run_rmse_uncertainty()
    run_xgboost_feature_importance()
    if not args.skip_horizons:
        run_horizon_evaluation()

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
