from torch import nn

import torch.optim as optim

from models.LSTMTransformer.StockDataLoader import create_dataloader

import torch

from trend.StockDatasetTrend import StockDatasetTrend
from trend.load_trend_model import load_trend_model
from trend.train_trend_model import train_trend_model
from trend.trend_parameter import get_trend_config


def training_trend_model(version="v1"):
    config = get_trend_config(version)
    # 获取模型参数
    mp = config.model_params
    tp = config.training_params
    dp = config.data_params
    Model = config.Model
    data_version = config.data

    # 检查设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Model(mp.input_dim, mp.hidden_dim, mp.num_layers, mp.num_heads).to(device)

    model = load_trend_model(model, tp.model_save_path)

    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=tp.learning_rate)

    dataset = StockDatasetTrend(feature_columns=dp.feature_columns, target_column=dp.target_column,
                                batch_size=tp.batch_size,
                                num_epochs=tp.num_epochs, data_version=data_version)
    dataloader = create_dataloader(dataset, tp.batch_size)

    # 使用训练数据训练模型
    train_trend_model(model, version, dataloader, criterion, optimizer)
