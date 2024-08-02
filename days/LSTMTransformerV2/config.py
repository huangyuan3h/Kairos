from days.LSTMTransformerV2.LSTMAttentionTransformerV2 import LSTMAttentionTransformerV2

config_LSTMAttentionTransformerV2_model = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 6,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 400,
    "model_save_path": "./model_files/days/LSTMAttentionTransformerV2_{}days.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformerV2,
    "data": "v2"
}