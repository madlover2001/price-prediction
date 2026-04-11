import pandas as pd

MESES_MAP = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12,
    "ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,
    "jul":7,"ago":8,"sep":9,"oct":10,"nov":11,"dic":12
}

def drop_unnamed(df):
    return df[[c for c in df.columns if not c.startswith("unnamed")]]

def crear_fecha(df):
    df = df.copy()
    df = drop_unnamed(df)

    df["mes"] = df["mes"].astype(str).str.lower().str.strip()
    df["mes_num"] = df["mes"].map(MESES_MAP)

    df["fecha"] = pd.to_datetime(
        df["año"].astype(str) + "-" + df["mes_num"].astype(int).astype(str) + "-01"
    )

    return df


# ================= IPC =================
def clean_ipc(df):
    df = crear_fecha(df)
    col = [c for c in df.columns if "ipc_alimentos" in c][0]
    return df[["fecha", col]].rename(columns={col: "ipc"})


# ================= AGRO =================
def clean_agro(df):
    df = crear_fecha(df)

    df = df[df["tipo_insumo"] == "FERTILIZANTES"]
    df["promedio_de_precio"] = pd.to_numeric(df["promedio_de_precio"], errors="coerce")

    return (
        df.groupby("fecha")["promedio_de_precio"]
        .mean()
        .reset_index()
        .rename(columns={"promedio_de_precio": "fertilizantes"})
    )


# ================= BANANO =================
def clean_banano(df):
    df = crear_fecha(df)

    df["precio_usdpresentación"] = pd.to_numeric(df["precio_usdpresentación"], errors="coerce")

    KG_40_LB = 18.1437
    df["precio_usd_kg"] = df["precio_usdpresentación"] / KG_40_LB

    return (
        df.groupby("fecha")["precio_usd_kg"]
        .mean()
        .reset_index()
        .rename(columns={"precio_usd_kg": "precio_banano_internacional"})
    )


# ================= PRODUCTOR =================
def clean_productor(df):
    df = crear_fecha(df)

    df["producto"] = df["producto"].astype(str).str.lower()
    df = df[df["producto"].str.contains("banano")]

    df["precio_promedio_usdkg"] = (
        df["precio_promedio_usdkg"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df["precio_promedio_usdkg"] = pd.to_numeric(
        df["precio_promedio_usdkg"], errors="coerce"
    )

    return (
        df.groupby("fecha")["precio_promedio_usdkg"]
        .mean()
        .reset_index()
        .rename(columns={"precio_promedio_usdkg": "precio_productor_banano"})
    )


# ================= IPP =================
def clean_ipp(df):

    df = df.copy()

    # limpiar columnas basura
    df = df[[c for c in df.columns if not c.startswith("unnamed")]]

    # limpiar nombres
    df.columns = df.columns.str.strip().str.lower()

    # 🔥 columnas fijas (ya conocidas)
    col_anio = "año" if "año" in df.columns else "anio"
    col_mes = "mes"
    col_ipp = "ippn"

    # limpiar mes
    df[col_mes] = df[col_mes].astype(str).str.lower().str.strip()

    MESES_MAP = {
        "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
        "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12
    }

    df["mes_num"] = df[col_mes].map(MESES_MAP)

    # crear fecha
    df["fecha"] = pd.to_datetime(
        df[col_anio].astype(str) + "-" + df["mes_num"].astype(int).astype(str) + "-01"
    )

    # limpiar valores
    df[col_ipp] = pd.to_numeric(df[col_ipp], errors="coerce")

    return df[["fecha", col_ipp]].rename(columns={col_ipp: "ipp"})