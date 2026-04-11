import matplotlib.pyplot as plt
import pandas as pd


def analyze_feature_importance(model, df, features):

    print("\n🔍 ANALISIS DE IMPORTANCIA DE VARIABLES")

    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n📊 Feature Importance:")
    print(feature_df)

    # ===============================
    # 📊 GRAFICO
    # ===============================
    plt.figure(figsize=(10,6))
    plt.barh(feature_df["feature"], feature_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Importancia de Variables (XGBoost)")
    plt.show()

    # ===============================
    # 💾 GUARDAR CSV
    # ===============================
    feature_df.to_csv("feature_importance.csv", index=False)

    print("\n💾 Feature importance guardado")