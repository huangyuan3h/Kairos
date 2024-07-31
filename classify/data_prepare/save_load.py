import os
import pickle
from typing import List, Tuple

import pandas as pd


def save_xy_data(x_data: List[pd.DataFrame], y_data: List[List[int]], output_dir: str):
    """将 x 和 y 数据分别保存到两个文件"""
    os.makedirs(output_dir, exist_ok=True)
    x_filepath = os.path.join(output_dir, "x_data.pkl")
    y_filepath = os.path.join(output_dir, "y_data.pkl")

    with open(x_filepath, 'wb') as f:
        pickle.dump(x_data, f)

    with open(y_filepath, 'wb') as f:
        pickle.dump(y_data, f)


def load_xy_data(input_dir: str) -> Tuple[List[pd.DataFrame], List[List[int]]]:
    """从文件中加载 x 和 y 数据"""
    x_filepath = os.path.join(input_dir, "x_data.pkl")
    y_filepath = os.path.join(input_dir, "y_data.pkl")

    with open(x_filepath, 'rb') as f:
        x_data = pickle.load(f)

    with open(y_filepath, 'rb') as f:
        y_data = pickle.load(f)

    return x_data, y_data
