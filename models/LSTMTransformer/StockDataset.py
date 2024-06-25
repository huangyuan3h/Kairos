import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np


class StockDataset(Dataset):
    def __init__(self, target_days, feature_columns, target_column):
        self.target_days = target_days
        self.feature_columns = feature_columns
        self.target_column = target_column

    def __len__(self):
        return 2 ** 32  # 一个很大的数字，表示理论上的无限样本

    def __getitem__(self, idx):
        data = self.get_data()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        x = data_scaled[:60, self.feature_columns]
        y = data_scaled[60:60 + self.target_days, self.target_column]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_data(self):
        # 模拟从某个数据源获取数据
        # 这里假设返回的是70行52列的随机浮点数据
        data = np.random.normal(size=(70, 52))
        if data.shape[0] >= 60 + self.target_days:
            return data
        else:
            raise ValueError("数据不足60行加上目标天数")