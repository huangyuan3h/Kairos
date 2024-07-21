from models.CNNLSTM.CNNLSTM import CNNLSTM

config_CNNLSTM = {
    "input_dim": 48,
    "hidden_dim": 512,
    "num_layers": 4,
    "num_heads": 60,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 5000,
    "model_save_path": "./model_files/CNN_LSTM.pth",
    "feature_columns": [i for i in range(48)],
    "target_column": "stock_close",
    "model": CNNLSTM
}