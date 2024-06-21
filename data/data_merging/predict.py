import pandas as pd

from data.raw import (
    get_stock_data_since,
    get_stock_profit_sheet_data,
    get_stock_balance_sheet_data,
    get_stock_cash_flow_sheet_data,
    get_sse_composite_index,
    get_szse_component_index,
    get_currency_exchange_rates,
)
from data.data_preprocessing import (
    clean_stock_data,
    clean_index_data,
    clean_currency_exchange_rates,
    merge_financial_data,
    clean_financial_data,
)

def get_stock_prediction_data(stock_code: str, start_date: str, n_days: int) -> pd.DataFrame:
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
        merged_data = pd.merge(cleaned_stock_data, cleaned_financial_data, on='report_date', how='left')
        merged_data = pd.merge(cleaned_stock_data, cleaned_currency_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_sse_index_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_szse_index_data, on='date', how='left')

        return merged_data
    except Exception as e:
        print(f"获取股票预测数据时发生错误：{e}")
        return None

stock_data = get_stock_prediction_data(stock_code='600000', start_date='20230101', n_days=30)
print(stock_data)