import torch
from torch.utils.data import IterableDataset

from models.LSTMTransformer.RandomStockData import RandomStockData

length_of_stock = 20


class StockDataset(IterableDataset):
    def __init__(self, feature_columns: list, target_column: str, batch_size: int, num_epochs: int):
        super(StockDataset).__init__()
        self.generate_pool = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        for i in range(length_of_stock):
            self.generate_pool.append(RandomStockData(feature_columns, target_column))

    def __iter__(self):
        for _ in range(self.batch_size):
            current_generator = self.generate_pool[0]
            x, y = current_generator.get_data()
            yield x, y
