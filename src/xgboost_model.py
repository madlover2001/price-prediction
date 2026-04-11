import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===============================
# 🔹 MÉTRICAS
# ===============================
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def directional_accuracy(y_true, y_pred):
    return np.mean(
        np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    )


# ===============================
# 🔹 DATA TABULAR
# ===============================
def prepare_tabular_data(df, feature_cols, target_col):

    df = df.sort_values("fecha").reset_index(drop=True)

    X = df[feature_cols].values
    y = df[target_col].values

    split = int(len(X) * 0.8)

    return X[:split], X[split:], y[:split], y[split:]


# ===============================
# 🔹 MODELO OPTIMIZADO
# ===============================
def build_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )


# ===============================
# 🔹 ENTRENAMIENTO
# ===============================
def train_xgboost(df, feature_cols, target_col, label):

    X_train, X_test, y_train, y_test = prepare_tabular_data(
        df, feature_cols, target_col
    )

    model = build_model()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ===============================
    # 🔹 MÉTRICAS
    # ===============================
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape_val = mape(y_test, y_pred)
    da = directional_accuracy(y_test, y_pred)

    print(f"\n📊 RESULTADOS XGBOOST {label}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    print(f"Directional Accuracy: {da:.2f}")

    return model, y_test, y_pred