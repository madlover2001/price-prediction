import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===============================
# 🔹 MÉTRICAS PRO
# ===============================
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(
        np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    )


# ===============================
# 🔹 SECUENCIAS
# ===============================
def create_sequences(data, target_index, window_size=8):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, target_index])

    return np.array(X), np.array(y)


# ===============================
# 🔹 PREPARACIÓN
# ===============================
def prepare_data(df, feature_cols, target_col):

    df = df.sort_values("fecha").reset_index(drop=True)

    data = df[feature_cols].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    target_index = feature_cols.index(target_col)

    X, y = create_sequences(data_scaled, target_index)

    split = int(len(X) * 0.8)

    return X[:split], X[split:], y[:split], y[split:], scaler


# ===============================
# 🔹 MODELO SIMPLE
# ===============================
def build_model(input_shape):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


# ===============================
# 🔹 ENTRENAMIENTO
# ===============================
def train_and_evaluate(df, feature_cols, target_col, label):

    X_train, X_test, y_train, y_test, scaler = prepare_data(df, feature_cols, target_col)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=4,
        verbose=1
    )

    y_pred = model.predict(X_test)

    # ===============================
    # 🔹 DESNORMALIZAR
    # ===============================
    y_test_inv, y_pred_inv = [], []

    for i in range(len(y_test)):
        dummy = np.zeros((1, len(feature_cols)))

        dummy[0, feature_cols.index(target_col)] = y_test[i]
        y_test_inv.append(
            scaler.inverse_transform(dummy)[0, feature_cols.index(target_col)]
        )

        dummy[0, feature_cols.index(target_col)] = y_pred[i]
        y_pred_inv.append(
            scaler.inverse_transform(dummy)[0, feature_cols.index(target_col)]
        )

    # ===============================
    # 🔹 MÉTRICAS
    # ===============================
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape_val = mape(y_test_inv, y_pred_inv)
    da = directional_accuracy(y_test_inv, y_pred_inv)

    print(f"\n📊 RESULTADOS {label}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    print(f"Directional Accuracy: {da:.2f}")

    return model, y_test_inv, y_pred_inv