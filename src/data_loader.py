import os
import pandas as pd

BASE_PATH = "outputs"

def normalizar_columnas(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df

def load_csv(filename):
    path = os.path.join(BASE_PATH, filename)
    df = pd.read_csv(path)
    return normalizar_columnas(df)

def load_all_data():
    return {
        "ipc": load_csv("ipc-alimentos-inflacion.xlsx_Inflación.csv"),
        "agro": load_csv("precios-agroquimicos-fertilizantes.xlsx_Hoja1.csv"),
        "banano": load_csv("precios-internacionales.xlsx_Banano.csv"),
        "productor": load_csv("precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv"),
        "ipp": load_csv("indices-sector.xlsx_IPP-N.csv"),
    }