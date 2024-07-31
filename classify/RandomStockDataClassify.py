from random import random

import numpy as np
import torch

from classify.get_classify_data import get_classify_xy_data_from_df
from data.data_merging.merge_data_v2 import get_random_v2_data

from models.standardize.FeatureStandardScaler import FeatureStandardScaler

learn_limit = 100

RANGE_SIZE = 70


def add_noise(x, noise_level=0.01):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise


class RandomStockDataClassify:

    def __init__(self, feature_columns: list, target_column: str, feature_scale: FeatureStandardScaler,
                 data_version="v1"):
        self.counter = 0
        self.data = get_random_v2_data()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scale = feature_scale
        self.data_version = data_version

    def get_data_inner(self):
        idx = self.counter
        range_data = self.data.loc[idx:RANGE_SIZE + idx - 1]
        x, y = get_classify_xy_data_from_df(range_data, self.feature_columns, self.target_column)

        self.counter = self.counter + 1
        if self.counter >= learn_limit:
            self.counter = 0
            self.data = get_random_v2_data()

        return x, y

    def get_data(self):
        x, y = self.get_data_inner()

        while y[0] == 1 and random() < 0.95:
            x, y = self.get_data_inner()

        # 添加噪声
        x = add_noise(x)
        x = torch.tensor(self.feature_scale.transform(x))
        y = torch.tensor(y)
        return x, y
