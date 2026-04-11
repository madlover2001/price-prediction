def analizar_dataframe(df, meta):
    print("\n" + "=" * 50)
    print(f"📂 Archivo: {meta['archivo']} | 📄 Hoja: {meta['hoja']}")
    print("=" * 50)

    print("\n📏 Shape:")
    print(df.shape)

    print("\n📌 Columnas:")
    print(df.columns.tolist())

    print("\n🔎 Info:")
    print(df.info())

    print("\n⚠️ Nulos:")
    print(df.isnull().sum())

    print("\n📊 Estadísticas:")
    print(df.describe(include="all"))