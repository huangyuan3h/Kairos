import torch


from data.data_merging.training_predict import get_random_v2_data_by_type
from days.get_days_data import get_xy_days_data_from_df

from models.standardize.FeatureStandardScaler import FeatureStandardScaler


learn_limit = 100

CACHE_LIMIT = 2

RANGE_SIZE = 70


class RandomStockDataDays:

    def __init__(self, feature_columns: list, target_column: str, feature_scale: FeatureStandardScaler, days=1):
        self.counter = 0
        self.cacheCounter = 0
        self.data = get_random_v2_data_by_type("training")
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scale = feature_scale
        self.days = days

    def get_data(self):
        idx = self.counter
        range_data = self.data.loc[idx:RANGE_SIZE + idx - 1]
        x, y = get_xy_days_data_from_df(range_data, self.feature_columns, self.target_column, self.days)
        x = torch.tensor(self.feature_scale.transform(x))
        y = torch.tensor(y)

        self.counter = self.counter + 1
        if self.counter >= learn_limit:
            self.counter = 0
            self.cacheCounter = self.cacheCounter + 1
        if self.cacheCounter >= CACHE_LIMIT:
            self.counter = 0
            self.cacheCounter = 0
            self.data = get_random_v2_data_by_type("training")

        return x, y
