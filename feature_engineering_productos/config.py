from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
FEATURE_OUTPUT_DIR = OUTPUTS_DIR / "feature_engineering_productos"

MARKETS_FILE = OUTPUTS_DIR / "precios-mercados-mayoristas-bodegas-comerciales.xlsx_Precios_Mercados12-25.csv"
PRODUCTOR_FILE = OUTPUTS_DIR / "precios-productor-ponderado.xlsx_TB_PPP_16_26_02_26.csv"
AGRO_FILE = OUTPUTS_DIR / "precios-agroquimicos-fertilizantes.xlsx_Hoja1.csv"
IPC_FILE = OUTPUTS_DIR / "ipc-alimentos-inflacion.xlsx_Inflación.csv"
IBC_FILE = OUTPUTS_DIR / "indices-sector.xlsx_IBC.csv"
IPM_FILE = OUTPUTS_DIR / "indices-sector.xlsx_IPM.csv"
IPPN_FILE = OUTPUTS_DIR / "indices-sector.xlsx_IPP-N.csv"

COMMON_START_DATE = "2016-01-01"

PRODUCT_CONFIGS = {
    "papa_superchola": {
        "product_name": "Papa Superchola",
        "product_key": "papa superchola",
        "output_dir": FEATURE_OUTPUT_DIR / "papa_superchola",
    },
    "tomate_rinon_invernadero": {
        "product_name": "Tomate Rinon De Invernadero",
        "product_key": "tomate rinon de invernadero",
        "output_dir": FEATURE_OUTPUT_DIR / "tomate_rinon_invernadero",
    },
    "maracuya": {
        "product_name": "Maracuya",
        "product_key": "maracuya",
        "output_dir": FEATURE_OUTPUT_DIR / "maracuya",
    },
}
