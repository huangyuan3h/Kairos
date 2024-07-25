
from models.SimpleLSTM_v2_1.SimpleLSTM import SimpleLSTM_v2_1

config_simple_lstm_v2_1 = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 2,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 6000,
    "model_save_path": "./model_files/simple_LSTM_v2_1.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": SimpleLSTM_v2_1,
    "data": "v2"
}
