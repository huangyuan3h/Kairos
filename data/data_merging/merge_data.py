import pandas as pd

import datetime
from datetime import date, timedelta
import random

from data.raw import (
    get_stock_data_since,
    get_stock_profit_sheet_data,
    get_stock_balance_sheet_data,
    get_stock_cash_flow_sheet_data,
    get_sse_composite_index,
    get_szse_component_index,
    get_currency_exchange_rates, get_random_code,
)
from data.data_preprocessing import (
    clean_stock_data,
    clean_index_data,
    clean_currency_exchange_rates,
    merge_financial_data,
    clean_financial_data,
)


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

    return df.join(interpolated_data.reset_index(drop=True), how='left')


def get_stock_total_data(stock_code: str, start_date: str, n_days: int) -> pd.DataFrame:
    """
    获取指定股票代码的预测数据，包含股票日线数据、财务数据、汇率数据和指数数据。

    Args:
        stock_code (str): 股票代码。
        start_date (str): 开始日期，格式为 'YYYYMMDD'。
        n_days (int): 获取的天数。

    Returns:
        pd.DataFrame: 包含所有数据的 DataFrame，如果获取失败则返回 None。
    """
    try:
        # 获取股票日线数据
        stock_data = get_stock_data_since(stock_code, start_date, n_days)
        if stock_data is None:
            return None
        cleaned_stock_data = clean_stock_data(stock_data.copy())

        # 获取财务数据
        profit_data = get_stock_profit_sheet_data(stock_code)
        balance_data = get_stock_balance_sheet_data(stock_code)
        cash_flow_data = get_stock_cash_flow_sheet_data(stock_code)
        if profit_data is None or balance_data is None or cash_flow_data is None:
            return None
        merged_financial_data = merge_financial_data(profit_data, balance_data, cash_flow_data)
        cleaned_financial_data = clean_financial_data(merged_financial_data.copy())

        # 获取汇率数据
        currency_data = get_currency_exchange_rates(start_date, n_days)
        if currency_data is None:
            return None
        cleaned_currency_data = clean_currency_exchange_rates(currency_data.copy())

        # 获取指数数据
        sse_index_data = get_sse_composite_index(start_date, n_days)
        szse_index_data = get_szse_component_index(start_date, n_days)
        if sse_index_data is None or szse_index_data is None:
            return None
        cleaned_sse_index_data = clean_index_data(sse_index_data.copy())
        cleaned_szse_index_data = clean_index_data(szse_index_data.copy())

        # 合并所有数据
        merged_data = pd.merge(cleaned_stock_data, cleaned_currency_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_sse_index_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_szse_index_data, on='date', how='left')
        merged_data = interpolate_financial_data(merged_data, cleaned_financial_data)

        return merged_data
    except Exception as e:
        print(f"获取股票预测数据时发生错误：{e}")
        return None


columns_to_remove = [
    'stock_code', 'year', 'quarter', 'date'
]


def drop_columns_and_reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """删除指定的列并重置索引.

    Args:
    df (pd.DataFrame): 输入的 DataFrame.
    columns_to_remove (list): 要删除的列名列表.

    Returns:
    pd.DataFrame: 删除指定列并重置索引后的 DataFrame.
    """
    df = df.drop(columns=columns_to_remove)
    df = df.reset_index(drop=True)
    return df


def get_random_available_date() -> str:
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    one_year_ago = today - timedelta(days=365)
    date_range = (five_years_ago, one_year_ago)
    random_days = random.randint(date_range[0].toordinal(), date_range[1].toordinal())
    random_date = date.fromordinal(random_days)
    return random_date.strftime("%Y%m%d")


def get_random_full_data() -> pd.DataFrame:
    result = None
    while result is None:
        code = get_random_code()
        start_date = get_random_available_date()
        result = get_stock_total_data(stock_code=code, start_date=start_date, n_days=365)
    return result


def get_random_valid_data() -> pd.DataFrame:
    stock_data = get_random_full_data()
    removed_data = drop_columns_and_reset_index(stock_data)

    return removed_data


# stock_data = get_stock_total_data(stock_code='600000', start_date='20230101', n_days=200)
#
# removed_data = drop_columns_and_reset_index(stock_data)
# print(removed_data)
