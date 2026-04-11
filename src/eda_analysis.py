import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def basic_info(df):
    print(df.info())
    print(df.describe())

def correlation(df):
    print("\n🔗 CORRELACIÓN")
    print(df.corr(numeric_only=True))

def plot_exogenous_scaled(df):
    cols = ["ipc", "fertilizantes", "precio_productor_banano", "ipp"]

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df_scaled[cols])

    plt.figure(figsize=(12,6))
    for col in cols:
        plt.plot(df_scaled["fecha"], df_scaled[col], label=col)

    plt.legend()
    plt.title("Variables exógenas (A++)")
    plt.show()