# Plan de Entrenamiento y Comparacion de Modelos

## Objetivo practico

Entrenar y comparar tres familias de modelos para prediccion de precios agropecuarios:

- ARIMA/SARIMAX
- LSTM
- XGBoost

Los modelos se evaluaran para tres productos:

- Papa Superchola
- Tomate Rinon de Invernadero
- Maracuya

La unidad final de prediccion sera `producto + provincia + fecha`. Esto permite que el futuro frontend seleccione un producto y una provincia para estimar el precio de mercado.

## Datasets de entrada

La fuente oficial para entrenamiento son los datasets ya generados en:

```text
outputs/feature_engineering_productos/
+-- papa_superchola/
|   +-- dataset_features.csv
|   +-- metadata_features.json
+-- tomate_rinon_invernadero/
|   +-- dataset_features.csv
|   +-- metadata_features.json
+-- maracuya/
    +-- dataset_features.csv
    +-- metadata_features.json
```

Cada `dataset_features.csv` contiene una serie temporal multivariada por provincia. Cada `metadata_features.json` define:

- producto
- columna objetivo
- columna de fecha
- provincias disponibles
- grupos de variables
- lista `recommended_model_features`

La variable objetivo oficial sera siempre:

```text
target_precio_mercado_usdkg
```

## Estructura del proyecto

Los entrenamientos se organizaran por familia de modelo para mantener separadas las dependencias y responsabilidades:

```text
training/
+-- common/
|   +-- data_loader.py
|   +-- metrics.py
|   +-- registry.py
|   +-- splits.py
+-- xgboost/
|   +-- config.py
|   +-- predict.py
|   +-- requirements.txt
|   +-- train.py
+-- lstm/
|   +-- config.py
|   +-- predict.py
|   +-- requirements.txt
|   +-- train.py
+-- arima/
|   +-- config.py
|   +-- predict.py
|   +-- requirements.txt
|   +-- train.py
+-- run_all.py
```

Los modelos entrenados se guardaran en:

```text
models/
+-- papa_superchola/
|   +-- arima/
|   +-- lstm/
|   +-- xgboost/
+-- tomate_rinon_invernadero/
|   +-- arima/
|   +-- lstm/
|   +-- xgboost/
+-- maracuya/
    +-- arima/
    +-- lstm/
    +-- xgboost/
```

Los resultados comparativos se guardaran en:

```text
outputs/model_results/
+-- metrics_summary.csv
+-- comparison_by_product.csv
+-- best_models.json
+-- predictions/
```

## Estrategia por modelo

### XGBoost

Se entrenara un modelo por producto usando todas las provincias disponibles. El modelo usara:

- `provincia_id`
- variables economicas exogenas
- variables de mercado
- variables temporales
- lags y rolling features

El split sera temporal por provincia: el 80% inicial de cada provincia se usara para entrenamiento y el 20% final para prueba.

### LSTM

Se entrenara un modelo por producto usando secuencias temporales por provincia. Las ventanas no deben mezclar provincias.

La preparacion de datos debe:

- ordenar por `provincia` y `fecha`
- crear ventanas de longitud fija por provincia
- entrenar el scaler solo con datos de entrenamiento
- evaluar en el tramo temporal final de cada provincia

### ARIMA/SARIMAX

Se ejecutara un job por producto, pero internamente se entrenara una serie por provincia. Esto se debe a que ARIMA/SARIMAX modela mejor una serie temporal univariada por entidad.

Para comparar contra LSTM y XGBoost, las metricas provinciales se consolidaran en una metrica agregada por producto.

## Metricas

Todos los modelos deben reportar:

- `MAE`
- `RMSE`
- `MAPE`
- `R2`
- `Directional Accuracy`

El criterio principal para seleccionar el mejor modelo por producto sera el menor `RMSE`. `MAE` y `MAPE` se usaran como soporte, y `Directional Accuracy` ayudara a evaluar si el modelo captura correctamente la direccion del cambio de precios.

## Artefactos esperados

Por cada combinacion `producto + modelo` se generara:

- modelo serializado
- metadata del entrenamiento
- metricas individuales
- predicciones sobre test
- configuracion utilizada
- lista de variables usadas

El archivo `outputs/model_results/best_models.json` sera el contrato inicial para el frontend. Debe indicar, por producto, el mejor modelo disponible y las rutas de sus artefactos.

## Validaciones

Antes y despues de entrenar se deben verificar estos puntos:

- los tres datasets existen
- cada dataset contiene `fecha`, `provincia`, `provincia_id`, `target_precio_mercado_usdkg`
- todas las variables de `recommended_model_features` existen en el dataset
- el split temporal mantiene test despues de train por provincia
- LSTM no mezcla ventanas entre provincias
- ARIMA consolida metricas provinciales por producto
- `metrics_summary.csv` contiene 9 filas cuando se ejecutan los tres modelos para los tres productos
- `best_models.json` contiene exactamente un mejor modelo por producto

## Supuestos

- La prediccion final sera por `producto + provincia`.
- Los datasets actuales de `outputs/feature_engineering_productos` son la fuente oficial para entrenamiento.
- Las dependencias se documentan por modelo porque LSTM y ARIMA requieren librerias adicionales.
- La primera version usa split temporal fijo `80/20`.
- Validacion cruzada temporal queda como mejora posterior.
