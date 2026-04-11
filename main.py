import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_all_data
from src.data_cleaning import *
from src.data_merge import merge_data
from src.eda_analysis import *
from src.xgboost_model import train_xgboost
from src.data_preparation_extra import add_lag_features
from src.model_monitoring import *
from src.feature_analysis import analyze_feature_importance

import matplotlib.pyplot as plt
import json


def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="Real")
    plt.plot(y_pred, label="Predicho")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def main():

    print("\n🚀 PIPELINE FINAL TESIS\n")

    # ===============================
    # 🔹 CARGA
    # ===============================
    data = load_all_data()

    # ===============================
    # 🔹 LIMPIEZA
    # ===============================
    datasets = {
        "ipc": clean_ipc(data["ipc"]),
        "agro": clean_agro(data["agro"]),
        "banano": clean_banano(data["banano"]),
        "productor": clean_productor(data["productor"]),
        "ipp": clean_ipp(data["ipp"]),
    }

    df = merge_data(datasets)

    # ===============================
    # 🔥 FEATURE ENGINEERING
    # ===============================
    df = add_lag_features(df)

    target = "precio_banano_internacional"

    features = [
        "precio_banano_internacional",
        "ipc",
        "fertilizantes",
        "precio_productor_banano",
        "ipp",
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
        "diff_1", "diff_2",
        "rolling_mean_3", "rolling_std_3"
    ]

    # ===============================
    # 💾 GUARDAR DATASETS
    # ===============================
    dataset_features = df[["fecha"] + features]

    dataset_features.to_csv("dataset_features_xgboost.csv", index=False)

    split = int(len(dataset_features) * 0.8)

    dataset_features.iloc[:split].to_csv("train_xgboost.csv", index=False)
    dataset_features.iloc[split:].to_csv("test_xgboost.csv", index=False)

    metadata = {
        "features": features,
        "target": target,
        "rows": len(df)
    }

    with open("metadata_modelo.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n💾 Datasets guardados correctamente")

    # ===============================
    # 📊 EDA
    # ===============================
    basic_info(df)
    correlation(df)

    # ===============================
    # ⚡ MODELO
    # ===============================
    model, y_test, y_pred = train_xgboost(
        df, features, target, "FINAL"
    )

    plot_predictions(y_test, y_pred, "XGBoost Final")

    # ===============================
    # 🔍 FEATURE IMPORTANCE + SHAP
    # ===============================
    analyze_feature_importance(model, df, features)

    # ===============================
    # 📊 MONITOREO
    # ===============================
    df_eval = prepare_evidently_data(df, y_test, y_pred)

    generate_drift_report(df_eval, df_eval)
    generate_performance_report(df_eval, df_eval)

    print("\n🏁 PROYECTO FINAL COMPLETADO\n")


if __name__ == "__main__":
    main()