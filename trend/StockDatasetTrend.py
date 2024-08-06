from torch.utils.data import IterableDataset

import random

from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from trend.RandomStockDataTrends import RandomStockDataTrends

length_of_stock = 128
steps_per_epoch = 1000


class StockDatasetTrend(IterableDataset):
    def __init__(self, feature_columns: list, target_column: str, batch_size: int, num_epochs: int, data_version="v1"):
        super(StockDatasetTrend).__init__()
        self.generate_pool = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_scaler = FeatureStandardScaler(data_version=data_version)
        for i in range(length_of_stock):
            self.generate_pool.append(RandomStockDataTrends(feature_columns, target_column, self.feature_scaler))

    def __iter__(self):
        for step in range(steps_per_epoch):
            for _ in range(self.batch_size):
                current_generator = random.choice(self.generate_pool)
                x, y = current_generator.get_data()
                yield x, y
