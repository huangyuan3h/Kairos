import pandas as pd
import numpy as np
from datetime import date, timedelta
from data.data_merging.merge_data import get_random_code, get_stock_total_data
import random
import datetime as dt

from models.standardize.FeatureStandardScaler import FeatureStandardScaler

times = 4000

record_day = 100


def get_random_record_num_available_date() -> dt.date:
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    end_day = today - timedelta(days=60)
    date_range = (five_years_ago, end_day)
    random_days = random.randint(date_range[0].toordinal(), date_range[1].toordinal())
    random_date = date.fromordinal(random_days)
    return random_date


def get_random_n_data() -> pd.DataFrame:
    result = None

    while result is None or len(result) <= record_day or np.isinf(result).any().any():
        code = get_random_code()
        start_date = get_random_record_num_available_date()
        end_date = start_date + timedelta(days=record_day * 2)

        result = get_stock_total_data(stock_code=code, start_date=start_date.strftime("%Y%m%d"),
                                      end_date=end_date.strftime("%Y%m%d"))
    return result


def build_df() -> pd.DataFrame:
    df_merged = None
    total_iterations = times
    print(f"开始构建数据帧，总共迭代 {total_iterations} 次")

    for i in range(times):
        if df_merged is None:
            df_merged = get_random_n_data()
            print(f"完成第 {i + 1} 次迭代，数据帧大小：{len(df_merged)}")
        else:
            to_append = get_random_n_data()
            df_merged = pd.concat([df_merged, to_append], ignore_index=True)
            print(f"完成第 {i + 1} 次迭代，数据帧大小：{len(df_merged)}")

    print(f"数据帧构建完成，最终大小：{len(df_merged)}")
    return df_merged


def fit_standard_scaler(df):
    feature_scaler = FeatureStandardScaler()
    feature_scaler.fit(df)
    feature_scaler.save_scaler()
