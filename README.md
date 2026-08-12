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
|       +-- predictions/                  # Predicciones base/full por modelo y producto
|       +-- horizon_predictions/          # Predicciones recursivas h=1,2,3 (config. full)
|       +-- plots/
+-- src/                                   # EDA y utilidades iniciales
+-- feature_engineering_productos/         # Pipeline de construcción de datasets
+-- training/                              # Entrenamiento ARIMA, LSTM y XGBoost
|   +-- common/                           # Splits, validación, ablación, incertidumbre, interpretabilidad
+-- models/                                # Modelos entrenados por producto/modelo/feature_set
+-- docs/                                  # Documentación técnica del entrenamiento y correcciones metodológicas
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

Cada `metadata_features.json` expone dos listas de variables usadas para el experimento de ablación (ver [Consideraciones](#consideraciones)):

- `base_model_features` (16 columnas): solo calendario, `provincia_id` y la historia propia del precio objetivo (lags/rolling/momentum de `target_precio_mercado_usdkg`).
- `full_model_features` (42 columnas): `base_model_features` + **únicamente** las 4 categorías de variables exógenas verdaderas que define la hipótesis (productor, fertilizantes, IPC/inflación, índices sectoriales), contemporáneas y rezagadas. Los 3 campos de contexto de mercado (`mercados_observaciones`, `mercados_distintos`, `tipos_mercado_distintos`) quedan documentados aparte en `feature_groups.market_context` de la metadata, pero no se usan como feature de modelado.

La imputación de variables exógenas usa únicamente propagación hacia adelante (`ffill`); no se usa `bfill` ni `interpolate()`, para evitar que un valor observado en un mes posterior se filtre hacia meses anteriores sin dato. Cada producto también expone un resumen agregado de preprocesamiento (registros iniciales/eliminados/imputados por fuente/finales) en `outputs/feature_engineering_productos/preprocessing_summary.csv`.

### `training/`

Pipeline de entrenamiento y comparación de modelos. La carpeta separa los entrenamientos por familia:

- `training/xgboost/`: entrenamiento tabular supervisado con variables exógenas, lags y una corrección de inercia usando `target_lag_1` (`blend`), cuyo peso se elige por validación entre `{0.0, 0.1, 0.2, 0.3}` junto con los hiperparámetros del modelo (no es un valor fijo).
- `training/lstm/`: entrenamiento secuencial con ventanas temporales por provincia, con validación interna propia y búsqueda de hiperparámetros (`window_size ∈ {6,8,10}` x `lstm_units ∈ {32,64}`).
- `training/arima/`: entrenamiento SARIMAX por provincia y consolidación de métricas por producto, en variante univariada (`base`) y con exógenas (`full`), usando las mismas ventanas de validación (por fecha, agrupadas) que XGBoost/LSTM.
- `training/common/`: carga de datasets, splits temporales/validación por ventanas, ablación (incluyendo bootstrap `base` vs `full`), ventana común de evaluación (test y validación), pronóstico multi-horizonte, incertidumbre estadística e interpretabilidad, y registro de artefactos.

Cada modelo se entrena en dos configuraciones de variables — `base` (solo historia propia del precio) y `full` (+ exógenas) — para poder atribuir el aporte de las exógenas por separado del efecto del algoritmo (ver [Consideraciones](#consideraciones)). La ejecución completa genera 18 entrenamientos: 3 productos x 3 modelos x 2 configuraciones.

```bash
python training/run_all.py --models xgboost arima lstm --product all --feature-sets base full --continue-on-error
```

También se puede entrenar cada familia por separado:

```bash
python training/xgboost/train.py --product all
python training/arima/train.py --product all
python training/lstm/train.py --product all
```

`run_all.py` también regenera, después de entrenar, la ventana común de validación, la ablación (incluyendo su significancia estadística), la ventana común de evaluación de test, la evaluación de horizontes 1-3 meses, los intervalos de confianza de RMSE y la interpretabilidad de XGBoost. Los resultados se guardan en:

- `models/<producto>/<modelo>/<base|full>/`: modelos serializados, metadata de entrenamiento y `validation_predictions.csv` (predicciones out-of-fold por fila, una por ventana de validación) por configuración.
- `outputs/model_results/metrics_summary.csv`: métricas de los 18 entrenamientos (test completo de cada modelo), incluyendo `validation_rmse_mean` y `operational_rmse_mean`.
- `outputs/model_results/ablation_summary.csv`: comparación `base` vs `full` por modelo y producto (aporte de las exógenas).
- `outputs/model_results/ablation_rmse_significance.csv`: significancia estadística de esa comparación (block bootstrap pareado + Holm-Bonferroni sobre las 9 combinaciones producto-modelo).
- `outputs/model_results/common_window_comparison.csv` / `common_validation_window.csv`: métricas de los 3 modelos calculadas sobre la misma intersección de fechas-provincia, en test y en validación respectivamente, para una comparación homogénea entre familias.
- `outputs/model_results/rmse_confidence_intervals.csv` / `rmse_pairwise_significance.csv`: incertidumbre de cada RMSE y significancia estadística de las diferencias entre modelos (block bootstrap + corrección de Holm-Bonferroni).
- `outputs/model_results/xgboost_feature_importance.csv`: importancia de variables por producto y configuración.
- `outputs/model_results/horizon_metrics.csv` y `horizon_predictions/`: métricas y predicciones para horizontes de 1, 2 y 3 meses.
- `outputs/model_results/comparison_by_product.csv` y `best_models.json`: mejor modelo por producto (configuración `full`, seleccionado por `operational_rmse_mean`: desempeño en pronóstico recursivo h=1,2,3 sobre la última ventana de validación, no por RMSE de un paso ni de test) — contrato inicial para consumo desde frontend.
- `outputs/model_results/predictions/`: predicciones sobre el conjunto de prueba, por producto/modelo/configuración.
- `outputs/model_results/plots/`: gráficas de valores reales vs predichos (configuración `full` por defecto; `python training/visualization/plot_predictions.py --feature-set base` para la línea base sin exógenas).

Metodología completa, incluyendo las correcciones aplicadas tras la revisión del docente y de un compañero, en [`docs/capitulo3_respuestas_entrenamiento.md`](docs/capitulo3_respuestas_entrenamiento.md), [`docs/correcciones_docente.md`](docs/correcciones_docente.md), [`docs/plan_maestro_companero.md`](docs/plan_maestro_companero.md) y [`docs/debate_revision_metodologica.md`](docs/debate_revision_metodologica.md).

## Entrenamiento y métricas

La partición de datos es un esquema híbrido, no un único split 80/20 (ver [Consideraciones](#consideraciones)):

- **80% desarrollo / 20% prueba**, con corte cronológico por provincia calculado una sola vez y compartido por los 3 modelos. El 20% de prueba se usa una sola vez, solo para reportar el resultado final.
- Dentro del 80% de desarrollo, **2-3 ventanas expansivas de validación** (cortes de fecha a nivel producto, compartidas por las 3 familias de modelo) sirven para seleccionar hiperparámetros/orden y para decidir la familia de modelo ganadora — el conjunto de prueba nunca participa en esa decisión. Cada modelo guarda sus predicciones fuera de muestra (OOF) de esas ventanas, lo que permite construir una ventana de validación realmente comparable entre modelos (`common_validation_window.csv`).

Los modelos se evalúan con las siguientes métricas:

- `MAE`: error absoluto medio. Se interpreta directamente en USD/kg. Por ejemplo, un MAE de `0.044` indica un error promedio aproximado de 4.4 centavos de USD por kg.
- `RMSE`: raíz del error cuadrático medio. Esta en USD/kg y penaliza mas los errores grandes.
- `MAPE`: error porcentual absoluto medio. Permite comunicar el error en porcentaje respecto al precio real.
- `R2`: proporción de variabilidad explicada por el modelo. Valores cercanos a 1 indican mejor ajuste; valores negativos indican mal desempeño.
- `Directional Accuracy`: porcentaje de veces que el modelo acierta la dirección de cambio del precio, es decir, si sube o baja.
- `n_test`: numero de observaciones usadas en el conjunto de prueba.
- `validation_rmse_mean`: promedio de RMSE de un paso entre las ventanas de validación internas del 80% de desarrollo. Se reporta, pero ya no decide la selección del modelo.
- `operational_rmse_mean` (nueva, decisiva): promedio de RMSE de un pronóstico **recursivo** h=1,2,3 (con las exógenas propagadas al último valor conocido), evaluado sobre la última ventana de validación. Es la métrica que decide qué modelo es "el mejor" por producto, porque refleja cómo se usa el modelo en producción (pronóstico multi-mes), no solo su error de un paso. El RMSE de test se sigue reportando, pero no participa en esa decisión.

## Resultados actuales

Configuración `full` (con variables exógenas), que es la que compite por producto. Tabla completa de las 18 combinaciones (`base`/`full` x 3 modelos x 3 productos) en `outputs/model_results/metrics_summary.csv`.

| Producto | Modelo | MAE | RMSE | MAPE (%) | R2 | Directional Accuracy | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maracuyá | XGBoost | 0.0758 | 0.0960 | 10.41 | 0.5695 | 0.8902 | 83 |
| Maracuyá | LSTM | 0.0979 | 0.1271 | 13.45 | 0.2458 | 0.6585 | 83 |
| Maracuyá | ARIMA (SARIMAX + exógenas) | 0.1031 | 0.1394 | 12.83 | 0.0930 | 0.6707 | 83 |
| Papa Superchola | XGBoost | 0.0457 | 0.0594 | 8.61 | 0.8365 | 0.8667 | 76 |
| Papa Superchola | LSTM | 0.0772 | 0.0972 | 18.39 | 0.5637 | 0.6892 | 75 |
| Papa Superchola | ARIMA (SARIMAX + exógenas) | 0.0606 | 0.0801 | 11.92 | 0.6818 | 0.8000 | 71 |
| Tomate Riñón de Invernadero | XGBoost | 0.0644 | 0.0804 | 10.74 | 0.7629 | 0.8557 | 98 |
| Tomate Riñón de Invernadero | LSTM | 0.0667 | 0.0825 | 11.35 | 0.7507 | 0.7320 | 98 |
| Tomate Riñón de Invernadero | ARIMA (SARIMAX + exógenas) | 0.0912 | 0.1074 | 14.00 | 0.5768 | 0.8144 | 98 |

Con el criterio de selección **operacional** (`operational_rmse_mean`: desempeño en pronóstico recursivo h=1,2,3, no RMSE de un paso ni de test — ver `best_models.json`), los mejores modelos actuales son:

| Producto | Mejor modelo | `operational_rmse_mean` | RMSE test | Interpretación |
|---|---:|---:|---:|---|
| Papa Superchola | XGBoost | 0.0632 | 0.0594 | Gana tanto en el criterio operacional como en RMSE de test. |
| Tomate Riñón de Invernadero | LSTM | 0.0831 | 0.0825 | Con el criterio de un paso XGBoost ganaba (ver tabla anterior), pero en pronóstico recursivo 1-3 meses LSTM tiene mejor desempeño en los 3 horizontes (`horizon_metrics.csv`), por eso es el seleccionado. |
| Maracuyá | XGBoost | 0.1578 | 0.0960 | Gana en ambos criterios; el aporte de las exógenas en este producto no es estadísticamente significativo (ver ablación abajo), así que la ventaja viene sobre todo del algoritmo, no de `full` frente a `base`. |

El criterio operacional puede elegir un modelo distinto al que gana en RMSE de test o de validación de un paso — es exactamente lo que ocurre en Tomate Riñón de Invernadero, donde LSTM resulta seleccionado pese a no tener el mejor RMSE de test de un paso, porque su desempeño en pronóstico multi-mes (que es como el prototipo se usa realmente) es mejor.

Para la comparación final se recomienda priorizar `operational_rmse_mean` para elegir modelo, reportar `MAE` para interpretación monetaria en USD/kg, usar `MAPE` para explicar el error porcentual, y citar `rmse_pairwise_significance.csv` / `ablation_rmse_significance.csv` antes de afirmar que un modelo (o la inclusión de exógenas) es mejor que la alternativa. `Directional Accuracy` debe presentarse como métrica complementaria para tendencia.

## Estado actual

Actualmente el repositorio contiene:

- preparación de datasets por producto, con dos configuraciones de variables (`base`/`full`, esta última limpia de campos de contexto de mercado) para poder medir el aporte real de las exógenas, más una tabla de cuantificación del preprocesamiento por producto
- entrenamiento de `XGBoost` (con blend `target_lag_1` tuneado por validación), `ARIMA/SARIMAX` (univariado y con exógenas, con ventanas de validación compartidas con los otros modelos) y `LSTM` (con búsqueda de hiperparámetros), con validación interna por ventanas expansivas y predicciones OOF por modelo
- comparación consolidada por producto sobre una ventana común de evaluación, tanto en test como en validación (mismas fechas-provincia para los 3 modelos)
- selección automática del mejor modelo según un criterio **operacional** (`operational_rmse_mean`: desempeño en pronóstico recursivo h=1,2,3), no según RMSE de un paso ni de test
- estimación de incertidumbre (block bootstrap + corrección de Holm-Bonferroni) tanto para diferencias entre modelos como para el aporte `base` vs `full`, e interpretabilidad de variables
- pronóstico recursivo a 1, 2 y 3 meses
- gráficas de precios reales vs predichos

## Consideraciones

- Los archivos fuente presentan variaciones de codificación y nombres de columnas, por lo que el proyecto incluye normalización explicita de encabezados, meses, provincias y productos.
- Algunas series no tienen cobertura completa hasta la misma fecha para todos los productos y provincias.
- En los datasets por producto se trabaja unicamente con provincias compartidas entre `mercados` y `productor`, para mantener consistencia en el cruce por fecha y territorio.
- ARIMA/SARIMAX se reporta en dos configuraciones: univariada (`base`), como línea base estadística clásica, y con exógenas (`full`), para poder aislar el aporte de las variables externas del efecto de cambiar de algoritmo. Ninguna de las dos usa overrides manuales de orden: el orden `(p,d,q)` se reoptimiza siempre mediante grid search validado, por separado para `base` y `full`, sobre las mismas ventanas que usan XGBoost y LSTM.
- El aporte de las variables exógenas **no es uniforme**: la ablación `base` vs `full` (`outputs/model_results/ablation_summary.csv`) muestra mejoras consistentes y estadísticamente significativas en Tomate Riñón de Invernadero (los 3 modelos) y en Papa Superchola para SARIMAX y XGBoost, y un efecto pequeño y no significativo en Maracuyá — 5 de las 9 combinaciones producto-modelo son significativas (`ablation_rmse_significance.csv`, block bootstrap + Holm-Bonferroni). Con solo 3 productos evaluados, este patrón es indicativo, no generalizable estadísticamente a otros productos.
- Se corrigieron dos fugas de información reales en la imputación/reconstrucción de variables exógenas: en `feature_engineering_productos/feature_builder.py` (`bfill`/`interpolate` reemplazados por `ffill` únicamente) y, en una ronda posterior, en `training/arima/train.py` (la misma fuga reintroducida al reconstruir las series por provincia). La segunda fue la corrección individual con mayor impacto de esta ronda: eliminó un colapso artificial de -105.7% en la ablación de SARIMAX para Papa Superchola, que pasó a ser una mejora real de +45.3%. Detalle en `docs/plan_maestro_companero.md`.
- Las diferencias de RMSE entre modelos, y entre `base` y `full`, se reportan con intervalo de confianza y corrección por comparaciones múltiples (block bootstrap por bloques, respetando la autocorrelación temporal dentro de cada provincia), no solo como valor puntual: algunas ventajas que parecen decisivas por RMSE no son estadísticamente significativas.
- Los horizontes de 2 y 3 meses asumen que las variables exógenas futuras se mantienen en su último valor conocido (carry-forward); el error crece con el horizonte, y esto debe leerse como un supuesto del prototipo, no como una predicción real de esas variables. El mismo supuesto se usa para el criterio operacional de selección de modelo.
- Limitaciones reconocidas y no resueltas: no se ejecutó validación espacial (*leave-one-province-out*) ni se generan intervalos predictivos (solo valores puntuales). Diseño técnico propuesto para ambas en `docs/debate_revision_metodologica.md`, secciones 8.b y 8.c, y consolidado en `docs/correcciones_docente.md`, sección "Limitaciones del estudio".
