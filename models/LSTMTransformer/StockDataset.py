import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data, target_days):
        self.data = data
        self.target_days = target_days
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(data)

    def __len__(self):
        return len(self.data) - self.target_days

    def __getitem__(self, idx):
        x = self.data_scaled[idx:idx+60]
        y = self.data_scaled[idx+60:idx+60+self.target_days, -1]  # 使用最后一列作为目标
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)