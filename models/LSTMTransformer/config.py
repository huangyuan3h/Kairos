from models.LSTMTransformer import LSTMAttentionTransformer

config_lstm_transformer_modelV1 = {
    "input_dim": 48,
    "hidden_dim": 512,
    "num_layers": 3,
    "num_heads": 16,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 10000,
    "model_save_path": "./model_files/lstm_transformer_model.pth",
    "feature_columns": [i for i in range(48)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformer
}