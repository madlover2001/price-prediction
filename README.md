# Price Prediction

Proyecto de tesis orientado a la preparación de datos, análisis exploratorio, entrenamiento y comparación de modelos para predicción de precios agropecuarios en Ecuador usando información del **SIPA (Sistema de Información Publica Agropecuaria)**.

El flujo actual trabaja con tres productos:

- `Papa Superchola`
- `Tomate Riñón de Invernadero`
- `Maracuyá`

La predicción se plantea a nivel `producto + provincia + fecha`, y la variable objetivo es el precio promedio de mercado en USD/kg.

## Objetivo

Construir datasets consistentes para predicción de precios, integrando:

- precios de mercado mayorista
- precios a nivel productor
- precios de fertilizantes
- IPC e inflación
- indices sectoriales agropecuarios

La variable objetivo en los datasets por producto es `target_precio_mercado_usdkg`.

## Fuentes de datos

Los archivos originales en formato `.xlsx` se descargaron desde el SIPA y se almacenan en la carpeta [`data`](./data). A partir de ellos se generaron archivos `.csv` depurados en [`outputs`](./outputs).

Fuentes utilizadas:

- `6. Precios productor`
- `7. Precios mercados mayoristas y bodegas comerciales`
- `11. Precios agroquímicos y fertilizantes`
- `12. IPC de alimentos e inflación`
- `13. Indices del sector`

Archivos relevantes en `outputs/`:

- `precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_Mercados12-25.csv`
- `precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv`
- `precios-agroquimicos-fertilizantes.xlsx_Hoja1.csv`
- `ipc-alimentos-inflacion.xlsx_Inflacion.csv`
- `indices-sector.xlsx_IBC.csv`
- `indices-sector.xlsx_IPM.csv`
- `indices-sector.xlsx_IPP-N.csv`

## Estructura del proyecto

```text
price-prediction/
+-- data/                                  # Archivos XLSX originales descargados del SIPA
+-- outputs/                               # CSV limpios, reportes y datasets generados
|   +-- eda_productos/
|   +-- eda_productor/
|   +-- ranking_productor_mercados/
|   +-- feature_engineering_productos/
|   +-- model_results/
+-- src/                                   # EDA y utilidades iniciales
+-- feature_engineering_productos/         # Pipeline de construcción de datasets
+-- training/                              # Entrenamiento ARIMA, LSTM y XGBoost
+-- models/                                # Modelos entrenados por producto/modelo
+-- docs/                                  # Documentación técnica del entrenamiento
+-- main_init.py                           # Conversion/EDA inicial de archivos
+-- requirements.txt
+-- README.md
```

## Componentes principales

### `main_init.py`

Pipeline inicial para cargar archivos originales, limpiar estructuras tabulares, ejecutar EDA básico y exportar resultados a CSV.

### `feature_engineering_productos/`

Pipeline especifico para construir datasets multivariados por `producto + provincia + fecha`.

Cada ejecutable produce:

- `dataset_features.csv`
- `metadata_features.json`

Ubicación:

- `outputs/feature_engineering_productos/papa_superchola/`
- `outputs/feature_engineering_productos/tomate_rinon_invernadero/`
- `outputs/feature_engineering_productos/maracuya/`

Cada fila del dataset representa una combinación de:

- `fecha`
- `provincia`
- `producto`

Variables principales incluidas:

- variable objetivo: `target_precio_mercado_usdkg`
- precio de mercado agregado por provincia y mes
- precio productor provincial y nacional
- agregados provinciales y nacionales de fertilizantes
- IPC de alimentos e inflación
- indices `IBC`, `IPM` e `IPP-N`
- variables de calendario
- indicadores de datos faltantes
- lags y rolling statistics por provincia

### `training/`

Pipeline de entrenamiento y comparación de modelos. La carpeta separa los entrenamientos por familia:

- `training/xgboost/`: entrenamiento tabular supervisado con variables exógenas, lags y una corrección de inercia usando `target_lag_1`.
- `training/lstm/`: entrenamiento secuencial con ventanas temporales por provincia.
- `training/arima/`: entrenamiento SARIMAX por provincia y consolidación de métricas por producto.
- `training/common/`: carga de datasets, splits temporales, métricas y registro de artefactos.

La ejecución completa genera 9 entrenamientos: 3 productos por 3 modelos.

```bash
python training/run_all.py --models xgboost arima lstm --product all --continue-on-error
```

También se puede entrenar cada familia por separado:

```bash
python training/xgboost/train.py --product all
python training/arima/train.py --product all
python training/lstm/train.py --product all
```

Los resultados se guardan en:

- `models/`: modelos serializados y metadata de entrenamiento.
- `outputs/model_results/metrics_summary.csv`: métricas de todos los entrenamientos.
- `outputs/model_results/comparison_by_product.csv`: mejor modelo por producto.
- `outputs/model_results/best_models.json`: contrato inicial para consumo desde frontend.
- `outputs/model_results/predictions/`: predicciones sobre el conjunto de prueba.
- `outputs/model_results/plots/`: gráficas de valores reales vs predichos.

## Entrenamiento y métricas

La partición de datos es temporal. Para cada provincia se usa el 80% inicial como entrenamiento y el 20% final como prueba. Esto evita que observaciones futuras entren al entrenamiento.

Los modelos se evalúan con las siguientes métricas:

- `MAE`: error absoluto medio. Se interpreta directamente en USD/kg. Por ejemplo, un MAE de `0.044` indica un error promedio aproximado de 4.4 centavos de USD por kg.
- `RMSE`: raíz del error cuadrático medio. Esta en USD/kg y penaliza mas los errores grandes. Es la métrica principal para seleccionar el mejor modelo.
- `MAPE`: error porcentual absoluto medio. Permite comunicar el error en porcentaje respecto al precio real.
- `R2`: proporción de variabilidad explicada por el modelo. Valores cercanos a 1 indican mejor ajuste; valores negativos indican mal desempeño.
- `Directional Accuracy`: porcentaje de veces que el modelo acierta la dirección de cambio del precio, es decir, si sube o baja.
- `n_test`: numero de observaciones usadas en el conjunto de prueba.

## Resultados actuales

| Producto | Modelo | MAE | RMSE | MAPE (%) | R2 | Directional Accuracy | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maracuyá | XGBoost | 0.0735 | 0.0949 | 10.06 | 0.5796 | 0.8780 | 83 |
| Maracuyá | LSTM | 0.0812 | 0.1048 | 10.17 | 0.4876 | 0.6585 | 83 |
| Maracuyá | ARIMA | 0.0885 | 0.1155 | 12.10 | 0.3838 | 0.5301 | 84 |
| Papa Superchola | XGBoost | 0.0448 | 0.0562 | 8.65 | 0.8538 | 0.9067 | 76 |
| Papa Superchola | LSTM | 0.0440 | 0.0566 | 8.86 | 0.8528 | 0.8493 | 74 |
| Papa Superchola | ARIMA | 0.1071 | 0.1208 | 22.26 | 0.2666 | 0.4930 | 72 |
| Tomate Riñón de Invernadero | LSTM | 0.0640 | 0.0782 | 9.56 | 0.7760 | 0.7629 | 98 |
| Tomate Riñón de Invernadero | XGBoost | 0.0644 | 0.0808 | 10.91 | 0.7608 | 0.8454 | 98 |
| Tomate Riñón de Invernadero | ARIMA | 0.1160 | 0.1432 | 17.03 | 0.2477 | 0.5464 | 98 |

Con el criterio principal de menor `RMSE`, los mejores modelos actuales son:

| Producto | Mejor modelo | RMSE | Interpretación |
|---|---:|---:|---|
| Maracuyá | XGBoost | 0.0949 | Mejor balance entre error, porcentaje y dirección de tendencia. |
| Papa Superchola | XGBoost | 0.0562 | Muy cercano a LSTM en error, pero con mejor Directional Accuracy. |
| Tomate Riñón de Invernadero | LSTM | 0.0782 | Menor RMSE y menor MAPE, aunque XGBoost acierta mejor la dirección. |

Para la comparación final se recomienda priorizar `RMSE`, reportar `MAE` para interpretación monetaria en USD/kg y usar `MAPE` para explicar el error porcentual. `Directional Accuracy` debe presentarse como métrica complementaria para tendencia.

## Estado actual

Actualmente el repositorio contiene:

- preparación de datasets por producto
- entrenamiento de `XGBoost`, `ARIMA/SARIMAX` y `LSTM`
- comparación consolidada por producto
- selección automática del mejor modelo según `RMSE`
- gráficas de precios reales vs predichos

## Consideraciones

- Los archivos fuente presentan variaciones de codificación y nombres de columnas, por lo que el proyecto incluye normalización explicita de encabezados, meses, provincias y productos.
- Algunas series no tienen cobertura completa hasta la misma fecha para todos los productos y provincias.
- En los datasets por producto se trabaja unicamente con provincias compartidas entre `mercados` y `productor`, para mantener consistencia en el cruce por fecha y territorio.
- ARIMA/SARIMAX se usa como linea base estadística por provincia; los modelos XGBoost y LSTM aprovechan variables exógenas y features temporales.
