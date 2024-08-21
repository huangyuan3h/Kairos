from torch import nn

import torch.optim as optim

from models.LSTMTransformer.StockDataLoader import create_dataloader

import torch

from operation.StockDatasetOperation import StockDatasetOperation
from operation.load_operation_model import load_operation_model
from operation.operation_parameter import get_operation_config
from operation.train_operation_model import train_operation_model


def training_operation_model(version="v1", days=1):
    config = get_operation_config(version)
    # 获取模型参数
    mp = config.model_params
    tp = config.training_params
    dp = config.data_params
    Model = config.Model
    data_version = config.data

    # 检查设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Model(mp.input_dim, mp.hidden_dim, mp.num_layers, mp.num_heads).to(device)

    model = load_operation_model(model, tp.model_save_path, days)

    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=tp.learning_rate)

    dataset = StockDatasetOperation(feature_columns=dp.feature_columns,
                                    batch_size=tp.batch_size,
                                    num_epochs=tp.num_epochs, data_version=data_version, days=days)
    dataloader = create_dataloader(dataset, tp.batch_size)

    # 使用训练数据训练模型
    train_operation_model(model, version, dataloader, criterion, optimizer, days)
