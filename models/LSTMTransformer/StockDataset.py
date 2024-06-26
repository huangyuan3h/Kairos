import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data.data_merging import get_random_valid_data

length_of_records = 100

def load_and_preprocess_data() -> torch.Tensor:
    data = get_random_valid_data()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 检查数据中是否有异常值
    if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
        raise ValueError("Data contains NaN or Inf values")

    return torch.tensor(data_scaled, dtype=torch.float32)


stock_list_num = 3


class StockDataset(Dataset):
    def __init__(self, target_days: int, feature_columns: list, target_column: int):
        self.target_days = target_days
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data = []
        for i in range(stock_list_num):
            self.data.append(load_and_preprocess_data())

    def __len__(self) -> int:
        return length_of_records * stock_list_num

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        id = idx % stock_list_num
        index = idx // stock_list_num

        current_data = self.data[id]

        # 确保索引不超出边界
        if index + 60 + 10 > len(current_data):
            raise IndexError("Index out of range")

        x = current_data[index:index + 60, self.feature_columns]
        # 计算1天、3天、5天和10天的涨跌幅均值
        y_1d = current_data[index + 60, self.target_column]
        y_3d = current_data[index + 60: index + 60 + 3, self.target_column].mean()
        y_5d = current_data[index + 60: index + 60 + 5, self.target_column].mean()
        y_10d = current_data[index + 60: index + 60 + 10, self.target_column].mean()

        y = torch.tensor([y_1d, y_3d, y_5d, y_10d])

        return x.clone().detach(), y.clone().detach()
