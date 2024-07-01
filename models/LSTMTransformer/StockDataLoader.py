import torch
import threading
import queue
from sklearn.preprocessing import StandardScaler
import numpy as np

from data.data_merging import get_random_valid_data

import time

length_of_records = 100


def _load_and_preprocess_data() -> torch.Tensor:
    """加载并预处理单个股票的数据"""
    data = get_random_valid_data()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 检查数据中是否有异常值
    if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
        raise ValueError("Data contains NaN or Inf values")

    return torch.tensor(data_scaled, dtype=torch.float32)


def _process_data(data: torch.Tensor, feature_columns: list, target_column: int, data_index: int):
    # 确保索引不超出边界
    if data_index + 60 + 10 > len(data):
        raise IndexError("Index out of range")

    x = data[data_index:data_index + 60, feature_columns]
    # 计算1天、3天、5天和10天的涨跌幅
    y_1d = data[data_index + 60, target_column]
    y_3d = data[data_index + 60: data_index + 60 + 3, target_column].sum()
    y_5d = data[data_index + 60: data_index + 60 + 5, target_column].sum()
    y_10d = data[data_index + 60: data_index + 60 + 10, target_column].sum()

    y = torch.tensor([y_1d, y_3d, y_5d, y_10d])

    return x.clone().detach(), y.clone().detach()


class StockDataLoader(torch.utils.data.DataLoader):
    def __init__(self, feature_columns: list, target_column: int, batch_size, shuffle=True):
        super().__init__(None, batch_size=batch_size, shuffle=shuffle)
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.queue = queue.Queue(maxsize=100)
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        while True:
            data = _load_and_preprocess_data()
            for data_index in range(length_of_records):
                features, target = _process_data(data, self.feature_columns, self.target_column, data_index)
                self.queue.put((features, target))

    def __len__(self):
        return length_of_records * 10000

    def __iter__(self):
        while True:
            if self.queue.empty():
                time.sleep(1)
                continue

            features, target = self.queue.get()
            yield features, target
