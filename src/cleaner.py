def limpiar_dataframe(df):
    # Normalizar columnas
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Eliminar duplicados
    df = df.drop_duplicates()

    return df