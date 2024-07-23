import datetime

import numpy as np
import pandas as pd

from data.data_merging.merge_data import get_stock_all_data, df_normalize_inf, get_random_code, \
    get_random_available_date, get_n_year_later, year
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

drop_str_columns = ['date', 'stock_code_left', 'stock_code_right']

column_to_keep = ['stock_close', 'stock_volume', 'stock_amplitude', 'stock_change_percent', 'rsi', 'ATR',
                  'KDJ_K',
                  'KDJ_D', 'KDJ_J', 'MACD', 'MACD_signal', 'MACD_hist', 'VWAP', 'month', 'day_of_week',
                  'is_end_of_week', 'is_end_of_month',
                  'Currency_EUR_CNY', 'Currency_EUR_CNY_MA_20', 'sse_open',
                  'sse_volume', 'sse_amplitude', 'sse_change_percent', 'sse_daily_return', 'sse_rsi', 'szse_volume',
                  'szse_amount', 'szse_amplitude', 'szse_change_percent', 'szse_daily_return', 'szse_rsi']



def keep_column_v2(df: pd.DataFrame):
    """
    保留 essential_features 和 consider_features 列，并将其转换为 float64 类型。

    Args:
        df (pd.DataFrame): 输入 DataFrame。

    Returns:
        pd.DataFrame: 处理后的 DataFrame。
    """
    if df is None:
        return None
    df = df[column_to_keep]
    df = df_normalize_inf(df)
    return df


def get_stock_v2_training_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票代码的预测数据，包含股票日线数据、财务数据、汇率数据和指数数据。

    Args:
        stock_code (str): 股票代码。
        start_date (str): 开始日期，格式为 'YYYYMMDD'。
        end_date (str): 开始日期，格式为 'YYYYMMDD'。

    Returns:
        pd.DataFrame: 包含所有数据的 DataFrame，如果获取失败则返回 None。
    """
    merged_data = get_stock_all_data(stock_code, start_date, end_date)
    final_df = keep_column_v2(merged_data)
    return final_df


def get_random_v2_data() -> pd.DataFrame:
    result = None

    while result is None or len(result) <= 200 * year:
        code = get_random_code()
        start_date = get_random_available_date()
        end_date = get_n_year_later(datetime.datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_v2_training_data(stock_code=code, start_date=start_date,
                                            end_date=end_date.strftime("%Y%m%d"))
    return result


def get_random_data_all() -> pd.DataFrame:
    result = None

    while result is None or len(result) <= 200 * year:
        code = get_random_code()
        start_date = get_random_available_date()
        end_date = get_n_year_later(datetime.datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_all_data(stock_code=code, start_date=start_date,
                                            end_date=end_date.strftime("%Y%m%d"))
    return result
