from __future__ import annotations

# Constantes compartidas por las 3 familias de modelos (xgboost, lstm, arima) para que la
# particion temporal y el criterio de seleccion de modelo sean consistentes entre ellas.
# Ver docs/correcciones_docente.md, puntos 2 y 3.

HOLDOUT_TEST_RATIO = 0.2
N_VALIDATION_WINDOWS = 3
MIN_VALIDATION_WINDOW_MONTHS = 6
FEATURE_SETS = ("base", "full")
MODEL_NAMES = ("xgboost", "lstm", "arima")
MIN_TEST_OBS_FOR_PROVINCE_CONCLUSIONS = 5
FORECAST_HORIZONS = (1, 2, 3)

# Tamano de bloque (en meses) para el block bootstrap de training/common/uncertainty.py.
# Los errores de un modelo en meses consecutivos de una misma provincia estan
# autocorrelacionados; remuestrear filas individuales i.i.d. subestima la varianza real.
BOOTSTRAP_BLOCK_SIZE_MONTHS = 4
