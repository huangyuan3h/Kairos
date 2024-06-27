from torch.utils.data import DataLoader
import torch

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel
from models.LSTMTransformer.StockDataset import StockDataset
import torch.nn as nn
import torch.optim as optim

from models.LSTMTransformer.load_model import load_model

from models.LSTMTransformer.train_model import train_model
from src.training.parameter import get_model_params, get_training_params, get_data_params


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    print(f"Average loss: {average_loss}")


def training():
    # 获取模型参数
    input_dim, hidden_dim, num_layers, num_heads, target_days = get_model_params()

    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads)

    # 获取训练参数
    batch_size, learning_rate, num_epochs, model_save_path = get_training_params()

    model = load_model(model, model_save_path)

    # 获取数据参数
    feature_columns, target_column = get_data_params()
    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = StockDataset(target_days, feature_columns, target_column)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    train_model(model, data_loader, criterion, optimizer, num_epochs, model_save_path)
    # 评估模型性能
    evaluate_model(model, data_loader, criterion)
