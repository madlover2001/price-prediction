MODEL_NAME = "lstm"
WINDOW_SIZE = 8

EPOCHS = 120
BATCH_SIZE = 8
LEARNING_RATE = 0.001
LSTM_UNITS = 64
RANDOM_SEED = 42
VERBOSE = 0
EARLY_STOPPING_PATIENCE = 15

# Busqueda de hiperparametros acotada (punto C.8 de la revision del companero): el tutor
# pidio un proceso reproducible para LSTM, igual que XGBoost/SARIMAX ya tienen. No se
# busca convertir la tesis en una investigacion de optimizacion de hiperparametros, asi
# que learning_rate/epochs/batch_size quedan fijos y solo se varian ventana temporal y
# capacidad de la capa LSTM, evaluados con las mismas ventanas de validacion expansivas.
HYPERPARAM_GRID = [
    {"window_size": 6, "lstm_units": 32},
    {"window_size": 6, "lstm_units": 64},
    {"window_size": 8, "lstm_units": 32},
    {"window_size": 8, "lstm_units": 64},
    {"window_size": 10, "lstm_units": 32},
    {"window_size": 10, "lstm_units": 64},
]
