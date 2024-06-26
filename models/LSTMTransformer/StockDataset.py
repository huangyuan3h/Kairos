import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data.data_merging import get_random_valid_data


def load_and_preprocess_data() -> torch.Tensor:
    data = get_random_valid_data()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 检查数据中是否有异常值
    if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
        raise ValueError("Data contains NaN or Inf values")

    return torch.tensor(data_scaled, dtype=torch.float32)


class StockDataset(Dataset):
    def __init__(self, target_days: int, feature_columns: list, target_column: int):
        self.target_days = target_days
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data = load_and_preprocess_data()

    def __len__(self) -> int:
        return len(self.data) - 60 - self.target_days

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        x = self.data[idx:idx + 60, self.feature_columns]
        y = self.data[idx + 60:idx + 60 + self.target_days, self.target_column]
        return x.clone().detach(), y.clone().detach()
