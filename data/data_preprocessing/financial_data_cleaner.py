import pandas as pd

# from data.raw import get_stock_profit_sheet_data, get_stock_balance_sheet_data, get_stock_cash_flow_sheet_data


def merge_financial_data(profit_data: pd.DataFrame, balance_data: pd.DataFrame, cash_flow_data: pd.DataFrame, stock_code:str) -> pd.DataFrame:
    """
    合并利润表、资产负债表和现金流量表数据。

    Args:
        profit_data (pd.DataFrame): 利润表数据，包含 'report_date', 'revenue', 'operating_profit', 'net_profit', 'basic_eps', 'year', 'quarter' 列。
        balance_data (pd.DataFrame): 资产负债表数据，包含 'report_date', 'total_assets', 'total_liabilities', 'total_equity', 'year', 'quarter' 列。
        cash_flow_data (pd.DataFrame): 现金流量表数据，包含 'report_date', 'net_cash_from_operating', 'net_cash_from_investing', 'net_cash_from_financing', 'net_increase_in_cce', 'year', 'quarter' 列。
        stock_code (str): stock code
    Returns:
        pd.DataFrame: 合并后的财务数据 DataFrame，包含所有列。
    """
    profit_data['report_date'] = pd.to_datetime(profit_data['report_date'], format='%Y%m%d')
    balance_data['report_date'] = pd.to_datetime(balance_data['report_date'], format='%Y%m%d')
    cash_flow_data['report_date'] = pd.to_datetime(cash_flow_data['report_date'], format='%Y%m%d')

    # 使用 merge 方法按 'report_date', 'year', 'quarter' 三列进行合并
    merged_data = pd.merge(profit_data, balance_data, on=['report_date', 'year', 'quarter'], how='left')
    merged_data = pd.merge(merged_data, cash_flow_data, on=['report_date', 'year', 'quarter'], how='left')

    # basic process
    columns_to_delete = ['year', 'quarter']
    merged_data = merged_data.drop(columns_to_delete, axis=1)
    merged_data['stock_code'] = stock_code

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
    merged_data = merged_data.fillna(method='ffill')  # 使用前值填充，可根据实际情况调整

    # 2. 处理异常值
    # 示例：处理毛利率异常值，使用中位数填充
    for col in ['revenue', 'total_operating_cost', 'operating_profit', 'gross_profit', 'net_profit',
                'basic_eps', 'rd_expenses', 'interest_income', 'interest_expense', 'investment_income',
                'cash_and_equivalents', 'accounts_receivable', 'inventory', 'net_fixed_assets',
                'short_term_borrowings', 'long_term_borrowings', 'total_equity', 'total_assets',
                'total_liabilities', 'net_cash_from_operating', 'net_cash_from_investing',
                'net_cash_from_financing', 'net_increase_in_cce', 'end_cash_and_cash_equivalents']:
        merged_data[col] = merged_data[col].where(merged_data[col] >= 0, merged_data[col].median())

    # 3. 特征工程
    # ... (计算各种财务比率，例如流动比率、资产负债率、净资产收益率等)

    # ----- 盈利能力 -----
    merged_data['gross_profit_margin'] = merged_data['gross_profit'] / merged_data['revenue']  # 毛利率
    merged_data['operating_profit_margin'] = merged_data['operating_profit'] / merged_data['revenue']  # 营业利润率
    merged_data['net_profit_margin'] = merged_data['net_profit'] / merged_data['revenue']  # 净利润率
    merged_data['return_on_equity'] = merged_data['net_profit'] / merged_data['total_equity']  # 净资产收益率 (ROE)
    merged_data['return_on_assets'] = merged_data['net_profit'] / merged_data['total_assets']  # 总资产收益率 (ROA)

    # ----- 营运能力 -----
    merged_data['asset_turnover'] = merged_data['revenue'] / merged_data['total_assets']  # 总资产周转率
    merged_data['inventory_turnover'] = merged_data['total_operating_cost'] / merged_data['inventory']  # 存货周转率
    merged_data['receivables_turnover'] = merged_data['revenue'] / merged_data['accounts_receivable']  # 应收账款周转率

    # ----- 偿债能力 -----
    merged_data['current_ratio'] = merged_data['total_assets'] / merged_data['total_liabilities']  # 流动比率
    merged_data['quick_ratio'] = (merged_data['total_assets'] - merged_data['inventory']) / merged_data[
        'total_liabilities']  # 速动比率
    merged_data['debt_to_asset_ratio'] = merged_data['total_liabilities'] / merged_data['total_assets']  # 负债资产比率
    # merged_data['interest_coverage_ratio'] = merged_data['operating_profit'] / merged_data[
    #     'interest_expense']  # 利息保障倍数

    # ----- 发展能力 -----
    # 需要根据多个报告期的财务数据计算同比增长率，这里仅作为示例
    merged_data['revenue_growth_rate'] = merged_data['revenue'].pct_change()  # 营业收入增长率
    merged_data['net_profit_growth_rate'] = merged_data['net_profit'].pct_change()  # 净利润增长率

    merged_data = merged_data.fillna(0)

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