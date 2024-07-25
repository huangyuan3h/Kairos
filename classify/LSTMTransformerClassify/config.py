from classify.LSTMTransformerClassify.LSTMTransformerModel import LSTMAttentionTransformerClassify


config_lstm_transformer_model_classify = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 1,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 7000,
    "model_save_path": "./model_files/lstm_transformer_model_classify.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": LSTMAttentionTransformerClassify,
    "data": "v2"
}