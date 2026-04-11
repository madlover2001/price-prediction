from src.loader import cargar_archivos
from src.cleaner import limpiar_dataframe
from src.eda import analizar_dataframe
from src.exporter import guardar_csv


def main():
    print("🚀 Iniciando EDA...\n")

    dataframes = cargar_archivos()

    for meta, df in dataframes:
        df = limpiar_dataframe(df)
        analizar_dataframe(df, meta)
        guardar_csv(df, meta)

    print("\n✅ EDA finalizado")


if __name__ == "__main__":
    main()