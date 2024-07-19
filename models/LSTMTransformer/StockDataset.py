from torch.utils.data import IterableDataset

from models.LSTMTransformer.RandomStockData import RandomStockData
import random

from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler

length_of_stock = 40


class StockDataset(IterableDataset):
    def __init__(self, feature_columns: list, target_column: str, batch_size: int, num_epochs: int):
        super(StockDataset).__init__()
        self.generate_pool = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_scaler = FeatureStandardScaler()
        self.target_scaler = TargetStandardScaler()
        for i in range(length_of_stock):
            self.generate_pool.append(RandomStockData(feature_columns, target_column,
                                                      self.feature_scaler, self.target_scaler))

    def __iter__(self):
        for _ in range(self.batch_size):
            current_generator = random.choice(self.generate_pool)
            x, y = current_generator.get_data()
            yield x, y
