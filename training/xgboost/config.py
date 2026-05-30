MODEL_NAME = "xgboost"
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.8
ENABLE_PARAM_TUNING = False
LAG_BLEND_FEATURE = "target_lag_1"
LAG_BLEND_WEIGHT = 0.2

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    **BASE_PARAMS,
}

PARAM_GRID = [
    PARAMS,
    {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.025,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_lambda": 1.5,
        **BASE_PARAMS,
    },
    {
        "n_estimators": 500,
        "max_depth": 2,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_lambda": 3.0,
        **BASE_PARAMS,
    },
    {
        "n_estimators": 350,
        "max_depth": 2,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 5.0,
        **BASE_PARAMS,
    },
    {
        "n_estimators": 900,
        "max_depth": 3,
        "learning_rate": 0.015,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_lambda": 2.0,
        **BASE_PARAMS,
    },
]
