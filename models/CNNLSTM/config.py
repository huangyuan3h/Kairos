from models.CNNLSTM.CNNLSTM import CNNLSTM

config_CNNLSTM = {
    "input_dim": 77,
    "hidden_dim": 768,
    "num_layers": 1,
    "num_heads": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 7000,
    "model_save_path": "./model_files/CNN_LSTM.pth",
    "feature_columns": [i for i in range(77)],
    "target_column": "stock_close",
    "model": CNNLSTM,
    "data": "v2"
}