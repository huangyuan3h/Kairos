
import torch

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel
from models.LSTMTransformer.StockDataLoader import StockDataLoader

import torch.nn as nn
import torch.optim as optim

from models.LSTMTransformer.load_model import load_model

from models.LSTMTransformer.train_model import train_model
from src.training.parameter import get_model_params, get_training_params, get_data_params
import itertools

num_eval_samples = 10000


def evaluate_model(model, data_loader, criterion, num_eval_samples: int):
    """
    评估模型性能

    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        num_eval_samples: 评估样本数量
    """
    model.eval()
    total_loss = 0
    num_processed = 0  # 记录已处理的样本数量

    with torch.no_grad():
        # 使用 itertools.islice 限制迭代次数
        for i, (inputs, targets) in enumerate(itertools.islice(data_loader, num_eval_samples)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_processed += inputs.shape[0]

        average_loss = total_loss / (i + 1) # 使用实际处理的batch数量计算平均损失
        print(f"Evaluated on {num_processed} samples. Average loss: {average_loss}")


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

    # 创建 StockDataLoader 实例
    train_data_loader = StockDataLoader(feature_columns=feature_columns, target_column=target_column,
                                        batch_size=batch_size, shuffle=True)

    # 使用训练数据训练模型
    train_model(model, train_data_loader, criterion, optimizer, num_epochs, model_save_path)

    # 创建用于评估的DataLoader (可以考虑使用不同的shuffle和batch_size)
    eval_data_loader = StockDataLoader(feature_columns=feature_columns, target_column=target_column,
                                       batch_size=batch_size, shuffle=False)

    # 评估模型性能
    evaluate_model(model, eval_data_loader, criterion, num_eval_samples)
