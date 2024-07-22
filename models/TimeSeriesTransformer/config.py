from models.TimeSeriesTransformer.TimeSeriesTransformer import TimeSeriesTransformer

config_TimeSeriesTransformer = {
    "input_dim": 48,  # 输入特征维度，保持不变
    "hidden_dim": 640,  # 降低隐藏层维度，可以根据需要调整
    "num_layers": 16,
    "num_heads": 64,  # 减少注意力头数，建议设置为 hidden_dim 的约数
    "batch_size": 128,
    "learning_rate": 1e-3,  # 降低学习率
    "num_epochs": 10000,  # 调整训练轮数，可以根据训练情况进行调整
    "model_save_path": "./model_files/TimeSeriesTransformer.pth",
    "feature_columns": [i for i in range(48)],
    "target_column": "stock_close",
    "model": TimeSeriesTransformer
}
