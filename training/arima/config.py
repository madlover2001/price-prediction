MODEL_NAME = "arima"
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.8

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 12)
MIN_OBSERVATIONS = 24

ORDER_GRID = [
    (1, 1, 1),
    (0, 1, 1),
    (1, 1, 0),
    (2, 1, 1),
]

SEASONAL_ORDER_GRID = [
    (0, 0, 0, 0),
    (1, 0, 1, 12),
]

PRODUCT_ORDER_OVERRIDES = {
    "papa_superchola": {
        "order": (1, 1, 0),
        "seasonal_order": (0, 0, 1, 12),
    },
    "tomate_rinon_invernadero": {
        "order": (1, 0, 0),
        "seasonal_order": (1, 0, 1, 12),
    },
    "maracuya": {
        "order": (1, 1, 1),
        "seasonal_order": (0, 0, 1, 12),
    },
}
