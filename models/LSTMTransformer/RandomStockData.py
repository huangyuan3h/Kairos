from data.data_merging.merge_data import get_random_valid_data
import torch

from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler

learn_limit = 100

x_row_num = 60

y_predict_day = 10


class RandomStockData:

    def __init__(self, feature_columns: list, target_column: str, feature_scale: FeatureStandardScaler,
                 target_scale: TargetStandardScaler):
        self.counter = 0
        self.data = get_random_valid_data()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scale = feature_scale
        self.target_scale = target_scale

    def get_data(self):
        df = self.data
        idx = self.counter
        data_to_scale = self.data.loc[idx + y_predict_day+1:x_row_num + idx + y_predict_day]
        scaled_data = self.feature_scale.transform(data_to_scale)

        x = torch.tensor(scaled_data)

        future_close = df[idx:idx + y_predict_day][self.target_column].values
        current_close = df[self.target_column].values[idx + y_predict_day]
        change_percentage = [(future_close[i - 1] - current_close) * 100 / current_close for i in [1, 6, 8, 10]]
        # 10 天， 5天， 3天， 1天
        scaled_change_percentage = self.target_scale.transform([change_percentage])[0]

        y = torch.tensor(scaled_change_percentage)
        self.counter = self.counter + 1
        if self.counter >= learn_limit:
            self.counter = 0
            self.data = get_random_valid_data()

        return x, y
