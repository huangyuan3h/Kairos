import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Tuple

from data.data_merging import get_random_valid_data

"""
同时训练20 个股票 100次

"""


length_of_records = 100



# get_random_full_data()


class StockDataset(IterableDataset):
    def __init__(self,  feature_columns: list, target_column: int):
        super(StockDataset).__init__()
        self.feature_columns = feature_columns
        self.target_column = target_column

    def __iter__(self):
        while True:


            seq = torch.tensor(self.df.iloc[start_idx:end_idx].values, dtype=torch.float32)
            targets = torch.tensor([self.calculate_target(end_idx, days) for days in self.target_days], dtype=torch.float32)
            yield seq, targets



    def getItem(self, idx: int, df: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        # 确保索引不超出边界
        if idx + 60 + 10 > len(df):
            raise IndexError("Index out of range")

        df_x = df[idx:idx + 60, self.feature_columns]
        x = torch.tensor(df_x.values)
        # 计算目标值（例如，未来 N 天的平均涨跌幅）
        future_close = df[idx + 60: idx + 60 + 10, self.target_column].values
        current_close = df[idx + 59, self.target_column]
        y = torch.tensor([(future_close[i-1] - current_close) / current_close for i in [1, 3, 5, 10]])

        return x, y
