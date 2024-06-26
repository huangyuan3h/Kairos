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

# 训练参数
batch_size = 32
learning_rate = 0.00001
num_epochs = 100
model_save_path = "../../model_files/lstm_transformer_model.pth"

# 数据参数
feature_columns = [i for i in range(input_dim)]
target_column = 1

# 接下来的训练次数
next_training_batch = 10


def main():
    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads)
    model = load_model(model, model_save_path)

    data = get_random_valid_data()
    predict_data = data[0:60]
    expected_data = data[60:70]
    scaler = StandardScaler()
    scaler.fit(predict_data)
    predictions = predict(model, predict_data, scaler, feature_columns)
    print("未来10天的预测数据：", predictions)
    print(expected_data["stock_change_percent"])


if __name__ == "__main__":
    main()
