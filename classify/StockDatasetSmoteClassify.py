import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
import pandas as pd
from typing import List, Tuple


class StockDatasetSmoteClassify(Dataset):
    def __init__(self, x_data: List[torch.Tensor], y_data: List[torch.Tensor]):
        """
        使用 SMOTE 进行数据增强的股票数据集类。

        Args:
            x_data (List[torch.Tensor]): 特征数据列表，每个元素为形状 (77, 60) 的张量。
            y_data (List[torch.Tensor]): 标签数据列表，每个元素为形状 (1,) 的张量。
        """
        # 将数据转换为 NumPy 数组，方便进行 SMOTE
        x_np = torch.stack(x_data).numpy()
        y_np = torch.cat(y_data).numpy()

        # 使用 SMOTE 进行过采样
        self.smote = SMOTE(random_state=42)
        x_resampled, y_resampled = self.smote.fit_resample(x_np.reshape(x_np.shape[0], -1), y_np)

        # 将过采样后的数据转换回张量
        self.x_resampled = torch.tensor(x_resampled, dtype=torch.float32).reshape(-1, 60, 77)
        self.y_resampled = torch.tensor(y_resampled, dtype=torch.long)

    def __len__(self):
        return len(self.x_resampled)

    def __getitem__(self, idx):
        x = self.x_resampled[idx]
        y = self.y_resampled[idx]

        return x, y
