import pandas as pd
from datetime import date, timedelta
from data.data_merging.merge_data import get_random_code
import random
import datetime as dt

from data.data_merging.merge_data_v2 import get_stock_v2_training_data
from models.LSTMTransformer.get_data import get_xy_data_from_df
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler
from src.training.parameter import get_config

times = 1000

record_day = 100

y_predict_day = 10


def get_random_record_num_available_date() -> dt.date:
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    end_day = today - timedelta(days=60)
    date_range = (five_years_ago, end_day)
    random_days = random.randint(date_range[0].toordinal(), date_range[1].toordinal())
    random_date = date.fromordinal(random_days)
    return random_date


def get_random_n_data(version="v1") -> pd.DataFrame:
    result = None

    while result is None or len(result) <= record_day:
        code = get_random_code()
        start_date = get_random_record_num_available_date()
        end_date = start_date + timedelta(days=record_day * 2)
        result = get_stock_v2_training_data(stock_code=code, start_date=start_date.strftime("%Y%m%d"),
                                                end_date=end_date.strftime("%Y%m%d"))

    return result


def calculate_change_percentages(df: pd.DataFrame, target_column: str, x_row_num: int) -> list:
    """
    计算涨幅百分比列表。

    Args:
        df (pd.DataFrame): 输入数据的 DataFrame。
        target_column (str): 目标变量的列名。
        x_row_num (int): 用于计算涨幅的行数。

    Returns:
        list: 涨幅百分比的列表。
    """
    change_percentage_list = []
    for idx in range(len(df) - x_row_num - y_predict_day):
        future_close = df[idx + x_row_num:idx + x_row_num + y_predict_day][target_column].values
        current_close = df[target_column].values[idx + x_row_num - 1]
        change_percentage = [(future_close[i - 1] - current_close) * 100 / current_close for i in [1, 3, 5, 10]]
        change_percentage_list.append(change_percentage)
    return change_percentage_list


def build_data(version="v1") -> (pd.DataFrame, list):
    df_merged = None
    change_percentage_list = []
    total_iterations = times
    config = get_config(version)
    dp = config.data_params
    print(f"开始构建数据帧，总共迭代 {total_iterations} 次")

    for i in range(times):
        df = get_random_n_data(config.data)
        df = df.head(70)
        x, y = get_xy_data_from_df(df, dp.feature_columns, dp.target_column)
        if df_merged is None:
            df_merged = x
            change_percentage_list.append(y)
            print(f"完成第 {i + 1} 次迭代，数据帧大小：{len(df_merged)}")
        else:
            to_append = x
            change_percentage_list.append(y)
            df_merged = pd.concat([df_merged, to_append], ignore_index=True)
            print(f"完成第 {i + 1} 次迭代，数据帧大小：{len(df_merged)}")

    print(f"数据帧构建完成，最终大小：{len(df_merged)}")
    return df_merged, change_percentage_list


def fit_feature_scaler(df, version="v1"):

    feature_scaler = FeatureStandardScaler(data_version=version)
    feature_scaler.fit(df)
    feature_scaler.save_scaler()


def fit_target_scaler(l: list, version="v1"):

    target_scaler = TargetStandardScaler(data_version=version)
    target_scaler.fit(l)
    target_scaler.save_scaler()
