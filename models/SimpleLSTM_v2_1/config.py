
from models.SimpleLSTM_v2_1.SimpleLSTM import SimpleLSTM_v2_1

config_simple_lstm_v2_1 = {
    "input_dim": 73,
    "hidden_dim": 512,
    "num_layers": 10,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-5,
    "num_epochs": 4000,
    "model_save_path": "./model_files/simple_LSTM_v2_1.pth",
    "feature_columns": [i for i in range(73)],
    "target_column": "stock_close",
    "model": SimpleLSTM_v2_1,
    "data": "v2"
}
