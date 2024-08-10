from operation.LSTMTransformer.LSTMTransformerModel import LSTMAttentionTransformer

config_lstm_transformer_operation_model = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 2,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 3000,
    "model_save_path": "./model_files/operation/lstm_transformer_model_{}days.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformer,
    "data": "v2"
}