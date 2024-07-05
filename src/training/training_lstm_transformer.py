from models.LSTMTransformer import StockDataset
from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel

import torch.nn as nn
import torch.optim as optim

from models.LSTMTransformer.StockDataLoader import create_dataloader
from models.LSTMTransformer.load_model import load_model

from models.LSTMTransformer.train_model import train_model
from src.training.parameter import get_model_params, get_training_params, get_data_params
import torch


def training():
    # 获取模型参数
    input_dim, hidden_dim, num_layers, num_heads = get_model_params()

    # 检查设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads).to(device)

    # 获取训练参数
    batch_size, learning_rate, num_epochs, model_save_path = get_training_params()

    model = load_model(model, model_save_path)

    # 获取数据参数
    feature_columns, target_column = get_data_params()

    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = StockDataset(feature_columns=feature_columns, target_column=target_column)
    dataloader = create_dataloader(dataset, batch_size)

    # 使用训练数据训练模型
    train_model(model, dataloader, criterion, optimizer, num_epochs, model_save_path)
