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

- `base_model_features`: solo calendario, `provincia_id` y la historia propia del precio objetivo (lags/rolling/momentum de `target_precio_mercado_usdkg`).
- `full_model_features`: `base_model_features` + variables exógenas verdaderas (productor, fertilizantes, IPC/inflación, índices sectoriales), contemporáneas y rezagadas.

La imputación de variables exógenas usa únicamente propagación hacia adelante (`ffill`); no se usa `bfill`, para evitar que un valor observado en un mes posterior se filtre hacia meses anteriores sin dato.

### `training/`

Pipeline de entrenamiento y comparación de modelos. La carpeta separa los entrenamientos por familia:

- `training/xgboost/`: entrenamiento tabular supervisado con variables exógenas, lags y una corrección de inercia usando `target_lag_1`.
- `training/lstm/`: entrenamiento secuencial con ventanas temporales por provincia, con validación interna propia.
- `training/arima/`: entrenamiento SARIMAX por provincia y consolidación de métricas por producto, en variante univariada (`base`) y con exógenas (`full`).
- `training/common/`: carga de datasets, splits temporales/validación por ventanas, ablación, ventana común de evaluación, pronóstico multi-horizonte, incertidumbre estadística e interpretabilidad, y registro de artefactos.

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

`run_all.py` también regenera, después de entrenar, la ablación, la ventana común de evaluación, la evaluación de horizontes 1-3 meses, los intervalos de confianza de RMSE y la interpretabilidad de XGBoost. Los resultados se guardan en:

- `models/<producto>/<modelo>/<base|full>/`: modelos serializados y metadata de entrenamiento por configuración.
- `outputs/model_results/metrics_summary.csv`: métricas de los 18 entrenamientos (test completo de cada modelo).
- `outputs/model_results/ablation_summary.csv`: comparación `base` vs `full` por modelo y producto (aporte de las exógenas).
- `outputs/model_results/common_window_comparison.csv`: métricas de los 3 modelos calculadas sobre la misma intersección de fechas-provincia, para una comparación homogénea.
- `outputs/model_results/rmse_confidence_intervals.csv` / `rmse_pairwise_significance.csv`: incertidumbre de cada RMSE y significancia estadística de las diferencias entre modelos (block bootstrap + corrección de Holm-Bonferroni).
- `outputs/model_results/xgboost_feature_importance.csv`: importancia de variables por producto y configuración.
- `outputs/model_results/horizon_metrics.csv` y `horizon_predictions/`: métricas y predicciones para horizontes de 1, 2 y 3 meses.
- `outputs/model_results/comparison_by_product.csv` y `best_models.json`: mejor modelo por producto (configuración `full`, seleccionado por RMSE de validación, no de test) — contrato inicial para consumo desde frontend.
- `outputs/model_results/predictions/`: predicciones sobre el conjunto de prueba, por producto/modelo/configuración.
- `outputs/model_results/plots/`: gráficas de valores reales vs predichos (configuración `full` por defecto; `python training/visualization/plot_predictions.py --feature-set base` para la línea base sin exógenas).

Metodología completa, incluyendo las correcciones aplicadas tras la revisión del docente, en [`docs/capitulo3_respuestas_entrenamiento.md`](docs/capitulo3_respuestas_entrenamiento.md), [`docs/correcciones_docente.md`](docs/correcciones_docente.md) y [`docs/debate_revision_metodologica.md`](docs/debate_revision_metodologica.md).

## Entrenamiento y métricas

La partición de datos es un esquema híbrido, no un único split 80/20 (ver [Consideraciones](#consideraciones)):

- **80% desarrollo / 20% prueba**, con corte cronológico por provincia calculado una sola vez y compartido por los 3 modelos. El 20% de prueba se usa una sola vez, solo para reportar el resultado final.
- Dentro del 80% de desarrollo, **2-3 ventanas expansivas de validación** (cortes de fecha a nivel producto) sirven para seleccionar hiperparámetros/orden y para decidir la familia de modelo ganadora — el conjunto de prueba nunca participa en esa decisión.

Los modelos se evalúan con las siguientes métricas:

- `MAE`: error absoluto medio. Se interpreta directamente en USD/kg. Por ejemplo, un MAE de `0.044` indica un error promedio aproximado de 4.4 centavos de USD por kg.
- `RMSE`: raíz del error cuadrático medio. Esta en USD/kg y penaliza mas los errores grandes.
- `MAPE`: error porcentual absoluto medio. Permite comunicar el error en porcentaje respecto al precio real.
- `R2`: proporción de variabilidad explicada por el modelo. Valores cercanos a 1 indican mejor ajuste; valores negativos indican mal desempeño.
- `Directional Accuracy`: porcentaje de veces que el modelo acierta la dirección de cambio del precio, es decir, si sube o baja.
- `n_test`: numero de observaciones usadas en el conjunto de prueba.
- `validation_rmse_mean` (nueva): promedio de RMSE entre las ventanas de validación internas del 80% de desarrollo. Es la métrica que decide qué modelo es "el mejor" por producto — el RMSE de test se reporta, pero no participa en esa decisión.

## Resultados actuales

Configuración `full` (con variables exógenas), que es la que compite por producto. Tabla completa de las 18 combinaciones (`base`/`full` x 3 modelos x 3 productos) en `outputs/model_results/metrics_summary.csv`.

| Producto | Modelo | MAE | RMSE | MAPE (%) | R2 | Directional Accuracy | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maracuyá | XGBoost | 0.0804 | 0.1015 | 11.03 | 0.5194 | 0.9024 | 83 |
| Maracuyá | LSTM | 0.1223 | 0.1432 | 16.99 | 0.0433 | 0.6585 | 83 |
| Maracuyá | ARIMA (SARIMAX + exógenas) | 0.1129 | 0.1535 | 13.87 | -0.1000 | 0.6707 | 83 |
| Papa Superchola | XGBoost | 0.0471 | 0.0626 | 8.84 | 0.8187 | 0.8667 | 76 |
| Papa Superchola | LSTM | 0.0758 | 0.1037 | 16.01 | 0.5034 | 0.8514 | 75 |
| Papa Superchola | ARIMA (SARIMAX + exógenas) | 0.1378 | 0.3333 | 35.39 | -4.5109 | 0.7429 | 71 |
| Tomate Riñón de Invernadero | XGBoost | 0.0607 | 0.0767 | 10.15 | 0.7842 | 0.8763 | 98 |
| Tomate Riñón de Invernadero | LSTM | 0.0624 | 0.0758 | 10.74 | 0.7894 | 0.8144 | 98 |
| Tomate Riñón de Invernadero | ARIMA (SARIMAX + exógenas) | 0.0651 | 0.0806 | 10.48 | 0.7618 | 0.8351 | 98 |

Con el criterio de selección corregido (menor `validation_rmse_mean`, no el RMSE de test), los mejores modelos actuales son:

| Producto | Mejor modelo | RMSE test | Interpretación |
|---|---:|---:|---|
| Maracuyá | XGBoost | 0.1015 | Gana en validación; la ventaja sobre ARIMA y LSTM es estadísticamente significativa (ver `rmse_pairwise_significance.csv`). |
| Papa Superchola | XGBoost | 0.0626 | Gana en validación y en test; la ventaja sobre ambos competidores es estadísticamente significativa. |
| Tomate Riñón de Invernadero | XGBoost | 0.0767 | Gana en validación, aunque el RMSE de test de los 3 modelos **no es estadísticamente distinguible entre sí** (ver punto 8 de `docs/correcciones_docente.md`). |

A diferencia de una versión anterior de este proyecto (donde el "mejor modelo" se elegía mirando el RMSE de test), con el criterio corregido XGBoost resulta seleccionado en los 3 productos. Esto no significa que sea siempre estadísticamente superior en test: en Tomate Riñón de Invernadero la diferencia con ARIMA y LSTM no es significativa, y se documenta así explícitamente en vez de presentarlo como una ventaja probada.

Para la comparación final se recomienda priorizar `validation_rmse_mean` para elegir modelo, reportar `MAE` para interpretación monetaria en USD/kg, usar `MAPE` para explicar el error porcentual, y citar `rmse_pairwise_significance.csv` antes de afirmar que un modelo es mejor que otro. `Directional Accuracy` debe presentarse como métrica complementaria para tendencia.

## Estado actual

Actualmente el repositorio contiene:

- preparación de datasets por producto, con dos configuraciones de variables (`base`/`full`) para poder medir el aporte real de las exógenas
- entrenamiento de `XGBoost`, `ARIMA/SARIMAX` (univariado y con exógenas) y `LSTM`, con validación interna por ventanas expansivas
- comparación consolidada por producto sobre una ventana común de evaluación (mismas fechas-provincia para los 3 modelos)
- selección automática del mejor modelo según `RMSE` de validación (no de test)
- estimación de incertidumbre (block bootstrap + corrección de Holm-Bonferroni) e interpretabilidad de variables
- pronóstico recursivo a 1, 2 y 3 meses
- gráficas de precios reales vs predichos

## Consideraciones

- Los archivos fuente presentan variaciones de codificación y nombres de columnas, por lo que el proyecto incluye normalización explicita de encabezados, meses, provincias y productos.
- Algunas series no tienen cobertura completa hasta la misma fecha para todos los productos y provincias.
- En los datasets por producto se trabaja unicamente con provincias compartidas entre `mercados` y `productor`, para mantener consistencia en el cruce por fecha y territorio.
- ARIMA/SARIMAX se reporta en dos configuraciones: univariada (`base`), como línea base estadística clásica, y con exógenas (`full`), para poder aislar el aporte de las variables externas del efecto de cambiar de algoritmo. Ninguna de las dos usa overrides manuales de orden: el orden `(p,d,q)` se reoptimiza siempre mediante grid search validado, por separado para `base` y `full`.
- El aporte de las variables exógenas **no es uniforme**: la ablación `base` vs `full` (`outputs/model_results/ablation_summary.csv`) muestra mejoras consistentes en Tomate Riñón de Invernadero, mejoras parciales en Papa Superchola y un efecto marginal o negativo en Maracuyá — con solo 3 productos evaluados, este patrón es indicativo, no generalizable estadísticamente a otros productos.
- Se corrigió una fuga de información real en la imputación de variables exógenas (`bfill` reemplazado por `ffill` únicamente); el detalle antes/después está en `docs/correcciones_docente.md`, punto 4.
- Las diferencias de RMSE entre modelos se reportan con intervalo de confianza y corrección por comparaciones múltiples, no solo como valor puntual: algunas ventajas que parecen decisivas por RMSE (p. ej. en Tomate Riñón de Invernadero) no son estadísticamente significativas.
- Los horizontes de 2 y 3 meses asumen que las variables exógenas futuras se mantienen en su último valor conocido (carry-forward); el error crece con el horizonte, y esto debe leerse como un supuesto del prototipo, no como una predicción real de esas variables.
- Limitaciones reconocidas y no resueltas: no se ejecutó validación espacial (*leave-one-province-out*) ni se generan intervalos predictivos (solo valores puntuales). Diseño técnico propuesto para ambas en `docs/debate_revision_metodologica.md`, secciones 8.b y 8.c, y consolidado en `docs/correcciones_docente.md`, sección "Limitaciones del estudio".
