from models.LSTMTransformerV0.LSTMTransformerModel import LSTMAttentionTransformerV0

config_lstm_transformer_modelV0 = {
    "input_dim": 48,
    "hidden_dim": 768,
    "num_layers": 4,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 10000,
    "model_save_path": "./model_files/lstm_transformer_model_v0.pth",
    "feature_columns": [i for i in range(48)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformerV0
}