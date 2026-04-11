import pandas as pd

def add_lag_features(df):

    df = df.sort_values("fecha").reset_index(drop=True)

    # ===============================
    # 🔹 LAGS (más memoria temporal)
    # ===============================
    for lag in range(1, 7):
        df[f"lag_{lag}"] = df["precio_banano_internacional"].shift(lag)

    # ===============================
    # 🔹 DIFERENCIAS (tendencia)
    # ===============================
    df["diff_1"] = df["precio_banano_internacional"].diff(1)
    df["diff_2"] = df["precio_banano_internacional"].diff(2)

    # ===============================
    # 🔹 ROLLING (suavizado)
    # ===============================
    df["rolling_mean_3"] = df["precio_banano_internacional"].rolling(3).mean()
    df["rolling_std_3"] = df["precio_banano_internacional"].rolling(3).std()

    # ===============================
    # 🔹 LIMPIEZA FINAL
    # ===============================
    df = df.dropna().reset_index(drop=True)

    return df