from days.LSTMTransformerV2.LSTMAttentionTransformerV2 import EnhancedLSTMAttentionTransformer

config_LSTMAttentionTransformerV2_model = {
    "input_dim": 77,
    "hidden_dim": 256,
    "num_layers": 3,
    "num_heads": 8,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_epochs": 2000,
    "model_save_path": "./model_files/days/LSTMAttentionTransformerV2_{}days.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": EnhancedLSTMAttentionTransformer,
    "data": "v2"
}