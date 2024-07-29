from classify.SimpleLSTMClassify.SimpleLSTMClassify import SimpleLSTMClassify

config_lstm_classify = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 2,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 4000,
    "model_save_path": "./model_files/lstm_classify.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": SimpleLSTMClassify,
    "data": "v2"
}