from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.data_merging import get_random_valid_data

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel
from models.LSTMTransformer.StockDataset import StockDataset
import torch.nn as nn
import torch.optim as optim

from models.LSTMTransformer.load_model import load_model
from models.LSTMTransformer.predict import predict
from models.LSTMTransformer.train_model import train_model

# 模型参数
input_dim = 47
hidden_dim = 128
num_layers = 3
num_heads = 8
target_days = 10

# 训练参数
batch_size = 32
learning_rate = 0.00001
num_epochs = 30
model_save_path = "../../model_files/lstm_transformer_model.pth"

# 数据参数
feature_columns = [i for i in range(input_dim)]
target_column = 7

# 接下来的训练次数
next_training_batch = 10


def main():
    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads)
    model = load_model(model, model_save_path)

    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(next_training_batch):
        dataset = StockDataset(target_days, feature_columns, target_column)
        data_loader = DataLoader(dataset, batch_size, shuffle=True)
        train_model(model, data_loader, criterion, optimizer, num_epochs, model_save_path)




if __name__ == "__main__":
    main()
