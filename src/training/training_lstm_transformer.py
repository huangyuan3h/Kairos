from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.data_merging import get_random_valid_data

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel
from models.LSTMTransformer.StockDataset import StockDataset
import torch.nn as nn
import torch.optim as optim

from models.LSTMTransformer.predict import predict
from models.LSTMTransformer.train_model import train_model

# 模型参数
input_dim = 47   # 输入特征维度
hidden_dim = 64  # LSTM隐藏层维度
num_layers = 2  # LSTM层数
num_heads = 4  # Transformer注意力头数
target_days = 10  # 预测未来天数

# 训练参数
batch_size = 32
learning_rate = 0.001
num_epochs = 100
model_save_path = "lstm_transformer_model.pth"

# 数据参数
feature_columns = [i for i in range(input_dim)]  # 特征列索引
target_column = 1  # 目标列索引


def main():
    dataset = StockDataset(target_days, feature_columns, target_column)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads, target_days)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, data_loader, criterion, optimizer, num_epochs, model_save_path)

    data = get_random_valid_data()
    predict_data = data[0:60]
    expected_data = data[60:70]
    scaler = StandardScaler()
    scaler.fit(predict_data)
    predictions = predict(model, predict_data, scaler, feature_columns)
    print("未来10天的预测数据：", predictions)
    print(expected_data)



if __name__ == "__main__":
    main()
