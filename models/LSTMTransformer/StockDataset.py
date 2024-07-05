import torch
from torch.utils.data import IterableDataset

from models.LSTMTransformer.RandomStockData import RandomStockData

"""
同时训练20 个股票数据生成器

"""
length_of_stock = 20



class StockDataset(IterableDataset):
    def __init__(self, feature_columns: list, target_column: str):
        super(StockDataset).__init__()
        self.generate_pool = []
        for i in range(length_of_stock):
            self.generate_pool.append(RandomStockData(feature_columns, target_column))

    def __iter__(self):
        while True:
            idx = torch.randint(0, length_of_stock, (1,)).item()
            current_generator = self.generate_pool[idx]
            x, y = current_generator.get_data()
            yield x, y
