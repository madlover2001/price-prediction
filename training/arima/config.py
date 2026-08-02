MODEL_NAME = "arima"

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
