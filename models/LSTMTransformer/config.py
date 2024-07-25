from models.LSTMTransformer import LSTMAttentionTransformer

config_lstm_transformer_modelV1 = {
    "input_dim": 77,
    "hidden_dim": 1024,
    "num_layers": 3,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 7000,
    "model_save_path": "./model_files/lstm_transformer_model.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformer,
    "data": "v2"
}