import datetime

import numpy as np
import pandas as pd

from data.data_merging.merge_data import get_stock_all_data, df_normalize_inf, get_random_code, \
    get_random_available_date, get_n_year_later, year
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


column_to_keep =  ['stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume', 'stock_amount',
        'stock_amplitude', 'stock_change_percent', 'stock_change', 'stock_turnover_rate', 'daily_return', 'ma5', 'ma20',
        'rsi', 'ATR', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist', 'VWAP', 'BOLL_mid',
        'BOLL_upper', 'BOLL_lower', 'day_of_week', 'month', 'quarter', 'is_end_of_week', 'is_end_of_month', 'Currency_USD_CNY',
        'Currency_EUR_CNY', 'Currency_USD_CNY_MA_5', 'Currency_USD_CNY_MA_20', 'Currency_EUR_CNY_MA_5', 'Currency_EUR_CNY_MA_20',
        'sse_open', 'sse_close', 'sse_high', 'sse_low', 'sse_change_percent',
        'sse_change', 'sse_turnover_rate', 'sse_daily_return', 'sse_ma5', 'sse_ma20', 'sse_rsi', 'szse_open', 'szse_close',
        'szse_high', 'szse_low', 'szse_change_percent', 'szse_change',
        'szse_turnover_rate', 'szse_daily_return', 'szse_ma5', 'szse_ma20', 'szse_rsi', 'rate_rate', 'rate_change',
        'qvix_open', 'qvix_high', 'qvix_low', 'qvix_close',  'gross_profit_margin', 'operating_profit_margin', 'net_profit_margin',
        'return_on_equity', 'return_on_assets', 'asset_turnover',
        'inventory_turnover', 'receivables_turnover',
        'current_ratio', 'quick_ratio', 'debt_to_asset_ratio', 'revenue_growth_rate', 'net_profit_growth_rate']




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
