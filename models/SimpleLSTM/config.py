from models.SimpleLSTM.SimpleLSTM import SimpleLSTM

config_simple_lstm = {
    "input_dim": 48,
    "hidden_dim": 512,
    "num_layers": 4,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 10000,
    "model_save_path": "./model_files/simple_LSTM.pth",
    "feature_columns": [i for i in range(48)],
    "target_column": "stock_close",
    "model": SimpleLSTM
}