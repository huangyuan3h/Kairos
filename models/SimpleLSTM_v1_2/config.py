
from models.SimpleLSTM_v1_2.SimpleLSTM import SimpleLSTM_v1_2

config_simple_lstm_v1_2 = {
    "input_dim": 48,
    "hidden_dim": 768,
    "num_layers": 10,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 4000,
    "model_save_path": "./model_files/simple_LSTM_v1_2.pth",
    "feature_columns": [i for i in range(48)],
    "target_column": "stock_close",
    "model": SimpleLSTM_v1_2
}