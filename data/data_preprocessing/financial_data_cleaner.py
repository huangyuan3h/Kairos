import pandas as pd

# from data.raw import get_stock_profit_sheet_data, get_stock_balance_sheet_data, get_stock_cash_flow_sheet_data


def merge_financial_data(profit_data: pd.DataFrame, balance_data: pd.DataFrame, cash_flow_data: pd.DataFrame) -> pd.DataFrame:
    """
    合并利润表、资产负债表和现金流量表数据。

    Args:
        profit_data (pd.DataFrame): 利润表数据，包含 'report_date', 'revenue', 'operating_profit', 'net_profit', 'net_profit_atsopc', 'basic_eps', 'year', 'quarter' 列。
        balance_data (pd.DataFrame): 资产负债表数据，包含 'report_date', 'total_assets', 'total_liabilities', 'total_equity', 'total_equity_atsopc', 'year', 'quarter' 列。
        cash_flow_data (pd.DataFrame): 现金流量表数据，包含 'report_date', 'net_cash_from_operating', 'net_cash_from_investing', 'net_cash_from_financing', 'net_increase_in_cce', 'year', 'quarter' 列。

    Returns:
        pd.DataFrame: 合并后的财务数据 DataFrame，包含所有列。
    """
    profit_data['report_date'] = pd.to_datetime(profit_data['report_date'], format='%Y%m%d')
    balance_data['report_date'] = pd.to_datetime(balance_data['report_date'], format='%Y%m%d')
    cash_flow_data['report_date'] = pd.to_datetime(cash_flow_data['report_date'], format='%Y%m%d')

    # 使用 merge 方法按 'report_date', 'year', 'quarter' 三列进行合并
    merged_data = pd.merge(profit_data, balance_data, on=['report_date', 'year', 'quarter'], how='left')
    merged_data = pd.merge(merged_data, cash_flow_data, on=['report_date', 'year', 'quarter'], how='left')

    return merged_data

def clean_financial_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    清洗合并后的财务数据。

    Args:
        merged_data (pd.DataFrame): 合并后的财务数据 DataFrame。

    Returns:
        pd.DataFrame: 清洗后的财务数据 DataFrame。
    """

    # 1. 处理缺失值
    # 使用前后数据填充
    merged_data = merged_data.infer_objects(copy=False)

    # 2. 处理异常值
    # 使用行业平均值或中位数替换明显不合理的财务数据
    # 这里需要根据具体的业务逻辑和数据情况进行判断
    merged_data['revenue'] = merged_data['revenue'].where(merged_data['revenue'] >= 0, merged_data['revenue'].mean())
    # 3. 特征工程
    # 计算各种财务比率，例如流动比率、资产负债率、净资产收益率等
    # 盈利能力
    merged_data['gross_profit_margin'] = merged_data['operating_profit'] / merged_data['revenue']  # 毛利率
    merged_data['net_profit_margin'] = merged_data['net_profit_atsopc'] / merged_data['revenue']  # 净利率
    merged_data['return_on_assets'] = merged_data['net_profit_atsopc'] / merged_data['total_assets']  # 资产收益率
    merged_data['return_on_equity'] = merged_data['net_profit_atsopc'] / merged_data['total_equity_atsopc']  # 净资产收益率
    merged_data['profit_on_operating_revenue'] = merged_data['operating_profit'] / merged_data['revenue']  # 营业利润率
    merged_data['net_profit_on_revenue'] = merged_data['net_profit'] / merged_data['revenue']  # 净利润率

    # 偿债能力
    merged_data['current_ratio'] = merged_data['total_assets'] / merged_data['total_liabilities']  # 流动比率
    merged_data['debt_to_asset_ratio'] = merged_data['total_liabilities'] / merged_data['total_assets']  # 负债资产比率

    # 成长性
    merged_data['net_profit_growth_rate'] = (merged_data['net_profit_atsopc'] - merged_data['net_profit_atsopc'].shift(
        1)) / merged_data['net_profit_atsopc'].shift(1)  # 归母净利润同比增长率

    return merged_data

# code = "600000"
#
# profit = get_stock_profit_sheet_data(code)
# balance = get_stock_balance_sheet_data(code)
# cash_flow = get_stock_cash_flow_sheet_data(code)
#
# merged = merge_financial_data(profit, balance, cash_flow)
#
# result = clean_financial_data(merged)
#
# print(result)