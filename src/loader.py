import pandas as pd
import os
from typing import List, Tuple, Dict, Any, Optional

from src.config import DATA_PATH


# 🔹 Palabras clave para ignorar hojas basura
HOJAS_IGNORAR = ["resumen", "dashboard", "grafico", "chart", "pivot", "indice"]


def es_hoja_valida(nombre_hoja: str) -> bool:
    """
    Filtra hojas irrelevantes
    """
    nombre = nombre_hoja.lower()
    return not any(palabra in nombre for palabra in HOJAS_IGNORAR)


def detectar_header(df_preview: pd.DataFrame) -> int:
    """
    Detecta automáticamente la fila del header
    basado en cantidad de valores no nulos
    """
    for i, row in df_preview.iterrows():
        if int(row.notnull().sum()) >= 3:
            return int(i) # type: ignore
    return 0


def validar_dataframe(df: pd.DataFrame) -> bool:
    """
    Verifica que el dataframe tenga datos útiles
    """
    if df.empty:
        return False

    if df.shape[1] < 2:
        return False

    return True


def limpiar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas
    """
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def leer_hoja(path: str, hoja: str) -> Optional[pd.DataFrame]:
    """
    Lee una hoja con detección automática de header
    """
    try:
        # 🔹 Preview para detectar header
        preview = pd.read_excel(
            path,
            sheet_name=hoja,
            header=None,
            nrows=20, 
            engine="openpyxl"
        )

        header_row: int = detectar_header(preview)

        # 🔹 Lectura real
        df = pd.read_excel(
            path,
            sheet_name=hoja,
            header=header_row
        )

        df = limpiar_columnas(df)

        if not validar_dataframe(df):
            return None

        return df

    except Exception as e:
        print(f"❌ Error leyendo hoja {hoja}: {e}")
        return None


def cargar_archivos() -> List[Tuple[Dict[str, Any], pd.DataFrame]]:
    """
    Carga todos los archivos Excel y sus hojas válidas
    """
    resultados: List[Tuple[Dict[str, Any], pd.DataFrame]] = []

    archivos = [f for f in os.listdir(DATA_PATH) if f.endswith(".xlsx")]

    if not archivos:
        print("⚠️ No se encontraron archivos .xlsx en la carpeta")

    for archivo in archivos:
        path = os.path.join(DATA_PATH, archivo)

        print(f"\n📂 Procesando archivo: {archivo}")

        try:
            xls = pd.ExcelFile(path)
        except Exception as e:
            print(f"❌ Error abriendo archivo {archivo}: {e}")
            continue

        for hoja in xls.sheet_names:

            if not es_hoja_valida(hoja): # type: ignore
                print(f"⏭️ Hoja ignorada: {hoja}")
                continue

            print(f"📄 Leyendo hoja: {hoja}")

            df = leer_hoja(path, hoja) # type: ignore

            if df is None:
                print(f"⚠️ Hoja descartada: {hoja}")
                continue

            meta = {
                "archivo": archivo,
                "hoja": hoja,
                "filas": df.shape[0],
                "columnas": df.shape[1],
            }

            resultados.append((meta, df))

    print(f"\n✅ Total dataframes cargados: {len(resultados)}")

    return resultados