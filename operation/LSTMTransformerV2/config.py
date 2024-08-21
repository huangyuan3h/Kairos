from operation.LSTMTransformerV2.LSTMAttentionTransformerV2 import LSTMAttentionTransformerV2

config_LSTMAttentionTransformerV2_model = {
    "input_dim": 77,
    "hidden_dim": 256,
    "num_layers": 3,
    "num_heads": 8,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "num_epochs": 2000,
    "model_save_path": "./model_files/operation/lstm_transformer_model_v2_{}days.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformerV2,
    "data": "v2"
}