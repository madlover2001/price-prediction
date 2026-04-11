import os

BASE_PATH = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = r"D:\IA - copia\data"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
REPORT_PATH = os.path.join(BASE_PATH, "reports")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)