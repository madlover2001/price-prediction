def merge_data(d):
    df = d["banano"]

    df = df.merge(d["ipc"], on="fecha", how="left")
    df = df.merge(d["agro"], on="fecha", how="left")
    df = df.merge(d["productor"], on="fecha", how="left")
    df = df.merge(d["ipp"], on="fecha", how="left")

    df = df.sort_values("fecha")

    # imputación
    df["ipc"] = df["ipc"].ffill()
    df["precio_productor_banano"] = df["precio_productor_banano"].ffill().bfill()
    df["ipp"] = df["ipp"].ffill().bfill()

    # 🔥 importante: alinear desde 2016
    df = df[df["fecha"] >= "2016-01-01"]

    return df