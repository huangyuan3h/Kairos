

from data.data_merging.merge_data import get_random_valid_data
import torch

from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler

learn_limit = 100

x_row_num = 60


class RandomStockData:

    def __init__(self, feature_columns: list, target_column: str, feature_scale: FeatureStandardScaler, target_scale: TargetStandardScaler):
        self.counter = 0
        self.data = get_random_valid_data()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scale = feature_scale
        self.target_scale = target_scale

    def get_data(self):
        df = self.data
        idx = self.counter
        data_to_scale = self.data.loc[self.counter:x_row_num + self.counter + 1]
        scaled_data = self.feature_scale.transform(data_to_scale)

        x = torch.tensor(scaled_data)

        future_close = df[idx + x_row_num:idx + x_row_num + 10][self.target_column].values
        current_close = df[self.target_column].values[idx + x_row_num - 1]
        change_percentage = [(future_close[i - 1] - current_close) * 100 / current_close for i in [1, 3, 5, 10]]
        scaled_change_percentage = self.target_scale.transform([change_percentage])[0]

        y = torch.tensor(scaled_change_percentage)
        self.counter = self.counter + 1
        if self.counter >= learn_limit:
            self.counter = 0
            self.data = get_random_valid_data()

        return x, y
