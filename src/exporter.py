import os
from src.config import OUTPUT_PATH


def guardar_csv(df, meta):
    nombre_archivo = f"{meta['archivo']}_{meta['hoja']}.csv"
    nombre_archivo = nombre_archivo.replace(" ", "_")

    path = os.path.join(OUTPUT_PATH, nombre_archivo)

    df.to_csv(path, index=False)

    print(f"💾 Guardado: {path}")