from data.data_merging.merge_data import get_random_valid_data
import torch

learn_limit = 500

x_row_num = 60


class RandomStockData:

    def __init__(self, feature_columns: list, target_column: str):
        self.counter = 0
        self.data = get_random_valid_data()
        self.tensor = torch.tensor(self.data.values)
        self.feature_columns = feature_columns
        self.target_column = target_column

    def get_data(self):
        df = self.data
        idx = self.counter
        x = self.tensor[self.counter: x_row_num + self.counter]

        future_close = df[idx + x_row_num:idx + x_row_num + 10][self.target_column].values
        current_close = df[self.target_column].values[idx + x_row_num - 1]
        change_percentage = [(future_close[i - 1] - current_close) * 100 / current_close for i in [1, 3, 5, 10]]
        y = torch.tensor(change_percentage)
        self.counter = self.counter + 1
        if self.counter >= learn_limit:
            self.counter = 0
            self.data = get_random_valid_data()
            self.tensor = torch.tensor(self.data.values)

        return x, y
