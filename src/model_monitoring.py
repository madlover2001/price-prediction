import pandas as pd
import numpy as np

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset


# ===============================
# 🔹 PREPARAR DATA
# ===============================
def prepare_evidently_data(df, y_true, y_pred):

    df = df.copy().iloc[-len(y_true):].reset_index(drop=True)

    df["target"] = y_true
    df["prediction"] = y_pred

    return df


# ===============================
# 🔹 DRIFT
# ===============================
def generate_drift_report(df_reference, df_current):

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=df_reference,
        current_data=df_current
    )

    report.save_html("reporte_drift.html")

    print("📊 Drift report generado")


# ===============================
# 🔹 PERFORMANCE
# ===============================
def generate_performance_report(df_reference, df_current):

    if len(df_current) < 5:
        print("⚠️ Muy pocos datos para reporte de performance, omitiendo...")
        return

    report = Report(metrics=[RegressionPreset()])

    report.run(
        reference_data=df_reference,
        current_data=df_current
    )

    report.save_html("reporte_modelo.html")
    print("📊 Performance report generado")