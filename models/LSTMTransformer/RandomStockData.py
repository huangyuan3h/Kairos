
import torch

from data.data_merging.merge_data_v2 import get_random_v2_data
from models.LSTMTransformer.get_data import get_xy_data_from_df
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler

learn_limit = 100

RANGE_SIZE = 70


class RandomStockData:

    def __init__(self, feature_columns: list, target_column: str, feature_scale: FeatureStandardScaler,
                 target_scale: TargetStandardScaler, data_version="v1"):
        self.counter = 0
        self.data = get_random_v2_data()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scale = feature_scale
        self.target_scale = target_scale
        self.data_version = data_version

    def get_data(self):
        idx = self.counter
        range_data = self.data.loc[idx:RANGE_SIZE + idx - 1]
        x, y = get_xy_data_from_df(range_data, self.feature_columns, self.target_column)
        x = torch.tensor(self.feature_scale.transform(x))
        y = torch.tensor(self.target_scale.transform([y])[0])

        self.counter = self.counter + 1
        if self.counter >= learn_limit:
            self.counter = 0
            self.data = get_random_v2_data()


        return x, y
