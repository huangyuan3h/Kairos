from torch.utils.data import IterableDataset

from models.LSTMTransformer.RandomStockData import RandomStockData
import random

length_of_stock = 60


class StockDataset(IterableDataset):
    def __init__(self, feature_columns: list, target_column: str, batch_size: int, num_epochs: int):
        super(StockDataset).__init__()
        self.generate_pool = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        for i in range(length_of_stock):
            self.generate_pool.append(RandomStockData(feature_columns, target_column))

    def __iter__(self):
        for epoch in range(self.num_epochs):
            # 每个 epoch 开始时打乱股票顺序
            random.shuffle(self.generate_pool)
            for batch in range(self.batch_size):
                # 随机选择一个股票
                current_generator = random.choice(self.generate_pool)
                x, y = current_generator.get_data()
                yield x, y
