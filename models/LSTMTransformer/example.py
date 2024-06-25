
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel
from models.LSTMTransformer.StockDataset import StockDataset
from models.LSTMTransformer.load_model import load_model
from models.LSTMTransformer.predict import predict
from models.LSTMTransformer.train_model import train_model
import torch

# 示例数据
data = np.random.normal(size=(1000, 52))  # 1000行，52列的浮点数据

# 参数设置
input_dim = len([i for i in range(52)])  # 使用所有列作为输入特征
hidden_dim = 64
num_layers = 2
num_heads = 4
target_days = 4  # 1天、3天、5天、10天
batch_size = 16
num_epochs = 50
learning_rate = 0.001
model_save_path = 'lstm_transformer_model.pth'
feature_columns = [i for i in range(52)]  # 使用所有列作为输入特征
target_column = 0  # 使用第0列作为目标预测列

# 数据集和数据加载器
dataset = StockDataset(target_days, feature_columns, target_column)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# 模型、损失函数和优化器
model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads, target_days)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型并保存
train_model(model, dataloader, criterion, optimizer, num_epochs, model_save_path)

# # 预测未来数据
# predictions = predict(model, data, dataset.get_scaler(), feature_columns)
# print("未来1、3、5、10天的预测数据：", predictions)
#
# # 加载模型并继续训练
# loaded_model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads, target_days)
# loaded_model = load_model(loaded_model, model_save_path)
#
# # 继续训练模型
# more_epochs = 20
# train_model(loaded_model, dataloader, criterion, optimizer, more_epochs, model_save_path)
#
# # 再次预测未来数据
# predictions_after_retraining = predict(loaded_model, data, dataset.get_scaler(), feature_columns)
# print("继续训练后的未来1、3、5、10天的预测数据：", predictions_after_retraining)