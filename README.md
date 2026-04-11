# Price Prediction

Proyecto de tesis orientado a la preparación de datos, análisis exploratorio y modelado de precios agropecuarios en Ecuador a partir de información del **SIPA (Sistema de Información Pública Agropecuaria)**.

El repositorio tiene dos líneas principales de trabajo:

- un pipeline de **feature engineering por producto** para `Papa Superchola`, `Tomate Riñón de Invernadero` y `Maracuyá`, pensado para alimentar modelos como `LSTM`, `ARIMA` y `XGBoost`

## Objetivo

Construir datasets consistentes para predicción de precios, integrando:

- precios de mercado mayorista
- precios a nivel productor
- precios de fertilizantes
- IPC e inflación
- índices sectoriales agropecuarios

La variable objetivo en los datasets por producto es el **precio de mercado en USD/kg** obtenido desde los registros de mercados mayoristas.

## Fuentes de datos

Los archivos originales en formato `.xlsx` se descargaron desde el SIPA y se almacenan en la carpeta [`data`](./data). A partir de ellos se generaron archivos `.csv` depurados en [`outputs`](./outputs).

Fuentes utilizadas:

- `6. Precios productor`
- `7. Precios mercados mayoristas y bodegas comerciales`
- `11. Precios agroquímicos y fertilizantes`
- `12. IPC de alimentos e inflación`
- `13. Índices del sector`

Archivos relevantes en `outputs/`:

- `precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_Mercados12-25.csv`
- `precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv`
- `precios-agroquimicos-fertilizantes.xlsx_Hoja1.csv`
- `ipc-alimentos-inflacion.xlsx_Inflación.csv`
- `indices-sector.xlsx_IBC.csv`
- `indices-sector.xlsx_IPM.csv`
- `indices-sector.xlsx_IPP-N.csv`

## Estructura del proyecto

```text
price-prediction/
├── data/                                  # Archivos XLSX originales descargados del SIPA
├── outputs/                               # CSV limpios, reportes y datasets generados
│   ├── eda_productos/
│   ├── eda_productor/
│   ├── ranking_productor_mercados/
│   └── feature_engineering_productos/
├── src/                                   # Pipeline original, EDA y utilidades de modelado
├── feature_engineering_productos/         # Pipeline nuevo por producto
├── main_init.py                           # Conversión/EDA inicial de archivos
├── requirements.txt
└── README.md
```

## Componentes principales

### `main_init.py`

Pipeline inicial para:

- cargar hojas y archivos originales
- limpiar estructuras tabulares
- ejecutar un EDA básico
- exportar resultados a CSV

Es la etapa usada para convertir los insumos iniciales hacia la carpeta `outputs/`.

### `feature_engineering_productos/`

Pipeline específico para construir datasets multivariados por `producto + provincia + fecha`.


## Datasets generados por producto

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

### Variables incluidas en los datasets por producto

Las variables derivadas incluyen, entre otras:

- variable objetivo: `target_precio_mercado_usdkg`
- precio de mercado agregado por provincia y mes
- precio productor provincial y nacional
- agregados provinciales y nacionales de fertilizantes
- IPC de alimentos e inflación
- índices `IBC`, `IPM` e `IPP-N`
- variables de calendario
- indicadores de datos faltantes
- lags y rolling statistics por provincia

Esto permite usar los datasets como base para modelos de series de tiempo y aprendizaje supervisado.


## Estado actual

Actualmente el repositorio contiene:

- preparación de datasets por producto para futuros entrenamientos con `LSTM`, `ARIMA` y `XGBoost`

## Consideraciones

- Los archivos fuente presentan variaciones de codificación y nombres de columnas, por lo que el proyecto incluye normalización explícita de encabezados, meses, provincias y productos.
- Algunas series no tienen cobertura completa hasta la misma fecha para todos los productos y provincias.
- En los datasets por producto se trabaja únicamente con provincias compartidas entre `mercados` y `productor`, para mantener consistencia en el cruce por fecha y territorio.
