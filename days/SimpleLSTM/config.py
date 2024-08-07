from days.SimpleLSTM.SimpleLSTM import SimpleLSTM

config_simple_lstm_days = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 2,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 3000,
    "model_save_path": "./model_files/days/simple_LSTM_{}days.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": SimpleLSTM,
    "data": "v2"
}
