from classify.SimpleLSTMClassify_v1_2.SimpleLSTMClassify import SimpleLSTMClassify_v1_2

config_lstm_classify_v1_2 = {
    "input_dim": 77,
    "hidden_dim": 256,
    "num_layers": 2,
    "num_heads": 8,
    "batch_size": 64,
    "learning_rate": 1e-5,
    "num_epochs": 1000,
    "model_save_path": "./model_files/lstm_classify_v1_2.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": SimpleLSTMClassify_v1_2,
    "data": "v2"
}