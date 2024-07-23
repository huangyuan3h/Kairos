import pandas as pd

import datetime
from datetime import date, timedelta
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

year = 1

from data.data_preprocessing import (
    clean_stock_data,
    clean_index_data,
    clean_currency_exchange_rates,
    clean_financial_data,
)
from db import get_db_session
from db.exchange_rate_daily import get_exchange_rate_by_date_range
from db.sh_index_daily import get_sh_index_daily_by_date_range
from db.stock_daily import get_stock_data_by_date_range
from db.stock_financial_data import get_financial_data_by_date_range
from db.stock_list import get_all_stock_list_data
from db.sz_index_daily import get_sz_index_daily_by_date_range

import numpy as np

essential_features = [
    'stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume',
    'stock_turnover_rate', 'ma5', 'ma20', 'rsi',
    'sse_open', 'sse_close', 'sse_high', 'sse_low', 'sse_volume',
    'sse_turnover_rate', 'sse_ma5', 'sse_ma20', 'sse_rsi',
    'szse_open', 'szse_close', 'szse_high', 'szse_low', 'szse_volume',
    'szse_turnover_rate', 'szse_ma5', 'szse_ma20', 'szse_rsi',
    'Currency_USD_CNY', 'Currency_EUR_CNY', 'Currency_USD_CNY_MA_5',
    'Currency_USD_CNY_MA_20', 'Currency_EUR_CNY_MA_5', 'Currency_EUR_CNY_MA_20'
]

consider_features = [
    'stock_amplitude', 'stock_change_percent', 'stock_change', 'stock_daily_return',
    'sse_amplitude', 'sse_change_percent', 'sse_change', 'sse_daily_return',
    'szse_amplitude', 'szse_change_percent', 'szse_change', 'szse_daily_return'
]

nonessential_features = [

    'revenue', 'total_operating_cost', 'operating_profit', 'gross_profit', 'net_profit',
    'basic_eps', 'rd_expenses', 'interest_income', 'interest_expense', 'investment_income',
    'cash_and_equivalents', 'accounts_receivable', 'inventory', 'net_fixed_assets',
    'short_term_borrowings', 'long_term_borrowings', 'total_equity', 'total_assets',
    'total_liabilities', 'net_cash_from_operating', 'net_cash_from_investing',
    'net_cash_from_financing', 'net_increase_in_cce', 'end_cash_and_cash_equivalents',
    'gross_profit_margin', 'operating_profit_margin', 'net_profit_margin', 'return_on_equity',
    'return_on_assets', 'asset_turnover', 'inventory_turnover', 'receivables_turnover',
    'current_ratio', 'quick_ratio', 'debt_to_asset_ratio', 'revenue_growth_rate',
    'net_profit_growth_rate'
]


def interpolate_financial_data(df: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    对财务数据进行线性插值，填充到日线数据中。

    Args:
        df (pd.DataFrame): 包含日线数据的 DataFrame，必须包含 'date' 列。
        financial_data (pd.DataFrame): 财务数据 DataFrame，必须包含 'report_date' 列。

    Returns:
        pd.DataFrame: 填充了插值后财务数据的 DataFrame。
    """
    # 将财务数据的 'report_date' 列设置为索引
    financial_data = financial_data.set_index('report_date')

    # 创建一个空 DataFrame 用于存储插值后的财务数据
    interpolated_data = pd.DataFrame(index=df.index, columns=financial_data.columns)

    # 遍历每条日线数据
    for i, row in df.iterrows():
        current_date = row['date']
        current_date = datetime.datetime.strptime(current_date, "%Y%m%d")
        # 找到当前日期所属的季度区间
        try:
            current_report_date = financial_data.index[financial_data.index < current_date][0]
        except IndexError:
            current_report_date = current_date

        quarter_data = financial_data.loc[current_report_date]

        interpolated_data.loc[i] = quarter_data

    merged_df = df.join(interpolated_data.reset_index(drop=True), how='left', lsuffix='_left', rsuffix='_right')
    return merged_df


def get_stock_all_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票代码的预测数据，包含股票日线数据、财务数据、汇率数据和指数数据。

    Args:
        stock_code (str): 股票代码。
        start_date (str): 开始日期，格式为 'YYYYMMDD'。
        end_date (str): 开始日期，格式为 'YYYYMMDD'。

    Returns:
        pd.DataFrame: 包含所有数据的 DataFrame，如果获取失败则返回 None。
    """
    try:
        # 获取股票日线数据
        with get_db_session() as db:
            stock_data = get_stock_data_by_date_range(db, stock_code, start_date, end_date)
        if stock_data is None or stock_data.empty:
            return None

        cleaned_stock_data = clean_stock_data(stock_data.copy())

        # 获取财务数据
        with get_db_session() as db:
            fin_data = get_financial_data_by_date_range(db, stock_code, start_date, end_date)
        if fin_data is None:
            return None

        cleaned_financial_data = clean_financial_data(fin_data.copy())

        # 获取汇率数据
        with get_db_session() as db:
            currency_data = get_exchange_rate_by_date_range(db, start_date, end_date)
        if currency_data is None:
            return None
        cleaned_currency_data = clean_currency_exchange_rates(currency_data.copy())

        # 获取指数数据
        with get_db_session() as db:
            sse_index_data = get_sh_index_daily_by_date_range(db, start_date, end_date)
        with get_db_session() as db:
            szse_index_data = get_sz_index_daily_by_date_range(db, start_date, end_date)
        if sse_index_data is None or szse_index_data is None:
            return None
        cleaned_sse_index_data = clean_index_data(sse_index_data.copy())
        cleaned_szse_index_data = clean_index_data(szse_index_data.copy())

        # 首先为每个数据框添加前缀
        cleaned_currency_data = cleaned_currency_data.add_prefix('Currency_')
        cleaned_sse_index_data = cleaned_sse_index_data.add_prefix('sse_')
        cleaned_szse_index_data = cleaned_szse_index_data.add_prefix('szse_')

        # 将 'date' 列名恢复为没有前缀的名称，以便进行合并
        cleaned_currency_data = cleaned_currency_data.rename(columns={'Currency_date': 'date'})
        cleaned_sse_index_data = cleaned_sse_index_data.rename(columns={'sse_date': 'date'})
        cleaned_szse_index_data = cleaned_szse_index_data.rename(columns={'szse_date': 'date'})

        # 合并所有数据
        merged_data = pd.merge(cleaned_stock_data, cleaned_currency_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_sse_index_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_szse_index_data, on='date', how='left')
        merged_data = interpolate_financial_data(merged_data, cleaned_financial_data)

        merged_data = merged_data.replace([np.inf, -np.inf], np.nan)
        merged_data = merged_data.ffill().bfill()
        return merged_data
    except Exception as e:
        print(f"Exception occurred in file: {e.__traceback__.tb_frame}")
        print(f"On line number: {e.__traceback__.tb_lineno}")
        print(stock_code, start_date, end_date)
        print(f"获取股票预测数据时发生错误：{e}")
        return None


def df_normalize_inf(df: pd.DataFrame) -> pd.DataFrame:
    # 将剩余列转换为 float64 类型
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"无法将列 '{col}' 转换为数字：{e}")
    return df


def get_stock_v1_training_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
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
    final_df = keep_columns_reset_type(merged_data)
    return final_df


def keep_columns_reset_type(df: pd.DataFrame):
    """
    保留 essential_features 和 consider_features 列，并将其转换为 float64 类型。

    Args:
        df (pd.DataFrame): 输入 DataFrame。

    Returns:
        pd.DataFrame: 处理后的 DataFrame。
    """
    # 保留 essential_features 和 consider_features 列
    if df is None:
        return None
    columns_to_keep = essential_features + consider_features
    df = df[columns_to_keep]
    df = df_normalize_inf(df)
    return df


def get_random_available_date() -> str:
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    one_year_ago = today - timedelta(days=365)
    date_range = (five_years_ago, one_year_ago)
    random_days = random.randint(date_range[0].toordinal(), date_range[1].toordinal())
    random_date = date.fromordinal(random_days)
    return random_date.strftime("%Y%m%d")


def get_random_code_from_df(df: pd.DataFrame) -> str:
    random_index = random.randint(0, len(df) - 1)
    random_row = df.iloc[random_index]
    random_code = random_row["code"]

    return random_code


def get_random_code() -> str:
    with get_db_session() as db:
        stock_list = get_all_stock_list_data(db)
    return get_random_code_from_df(stock_list)


def get_n_year_later(dt):
    """
    获取指定日期一年后的日期

    Args:
        dt: datetime 对象

    Returns:
        datetime 对象，代表指定日期一年后的日期
    """
    if not isinstance(dt, datetime.datetime):
        raise TypeError("参数 dt 必须为 datetime 对象")

    next_day = dt + datetime.timedelta(days=365 * year)
    return next_day


def get_random_v1_data() -> pd.DataFrame:
    result = None

    while result is None or len(result) <= 200 * year or np.isinf(result).any().any():
        code = get_random_code()
        start_date = get_random_available_date()
        end_date = get_n_year_later(datetime.datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_v1_training_data(stock_code=code, start_date=start_date,
                                            end_date=end_date.strftime("%Y%m%d"))
    return result


def get_random_valid_data() -> pd.DataFrame:
    df = get_random_v1_data()

    return df

# stock_data = get_stock_total_data(stock_code='600000', start_date='20220101', n_days=200)
#
# removed_data = drop_columns_and_reset_index(stock_data)
# print(removed_data)
