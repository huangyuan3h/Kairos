import akshare as ak
import pandas as pd


def add_missing_columns(df, cols_to_add):
    """
    向DataFrame中添加缺失的列

    Args:
      df: 输入的DataFrame
      cols_to_add: 要添加的列名列表

    Returns:
      添加列后的DataFrame
    """

    for col in cols_to_add:
        if col not in df.columns:
            df[col] = [None] * len(df)
    return df


def get_stock_profit_sheet_data(stock_code: str) -> pd.DataFrame:
    """
    获取指定股票代码的所有年份和季度的利润表数据，保留重要列并转换为英文列名。

    Args:
        stock_code (str): 股票代码，例如 '600000'。

    Returns:
        pd.DataFrame: 包含利润表数据的 DataFrame，如果获取失败则返回 None。
                      DataFrame 包含以下英文列名：
                          - report_date: 报告日期
                          - revenue: 营业收入
                          - operating_profit: 营业利润
                          - net_profit: 净利润
                          - basic_eps: 基本每股收益
                          - year: 年份
                          - quarter: 季度
    """

    try:
        profit_data = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")

        # 检查是否成功获取数据
        if profit_data is None or profit_data.empty:
            print(f"未能获取股票 {stock_code} 的利润表数据，请检查股票代码是否正确。")
            return None

        # 提取年份和季度信息
        profit_data['year'] = profit_data['报告日'].str[:4].astype(int)
        profit_data['quarter'] = profit_data['报告日'].str[4:6].astype(int) // 3

        keep_list = [
            '报告日', '营业收入', '营业总成本', '营业利润', '利润总额', '净利润', '基本每股收益',
            '研发费用', '利息收入', '利息支出', '投资收益', 'year', 'quarter'
        ]

        add_missing_columns(profit_data, keep_list)

        # 保留重要列并重命名
        profit_data = profit_data[keep_list]

        profit_data.rename(columns={
            '报告日': 'report_date',
            '营业收入': 'revenue',
            '营业总成本': 'total_operating_cost',
            '营业利润': 'operating_profit',
            '利润总额': 'gross_profit',
            '净利润': 'net_profit',
            '基本每股收益': 'basic_eps',
            '研发费用': 'rd_expenses',
            '利息收入': 'interest_income',
            '利息支出': 'interest_expense',
            '投资收益': 'investment_income'
        }, inplace=True)

        return profit_data

    except Exception as e:
        print(f"获取利润表数据时发生错误：{e} stock_code：{stock_code}")
        return None


def get_stock_balance_sheet_data(stock_code: str) -> pd.DataFrame:
    """
    获取指定股票代码的所有年份和季度的资产负债表数据，保留重要列并转换为英文列名。

    Args:
        stock_code (str): 股票代码，例如 '600000'。

    Returns:
        pd.DataFrame: 包含资产负债表数据的 DataFrame，如果获取失败则返回 None。
                      DataFrame 包含以下英文列名：
                          - report_date: 报告日期
                          - total_assets: 资产总计
                          - total_liabilities: 负债合计
                          - total_equity: 股东权益合计
                          - year: 年份
                          - quarter: 季度
    """

    try:
        balance_data = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")

        # 检查是否成功获取数据
        if balance_data is None or balance_data.empty:
            print(f"未能获取股票 {stock_code} 的资产负债表数据，请检查股票代码是否正确。")
            return None

        # 提取年份和季度信息
        balance_data['year'] = balance_data['报告日'].str[:4].astype(int)
        balance_data['quarter'] = balance_data['报告日'].str[4:6].astype(int) // 3

        keep_list = [
            '报告日', '货币资金', '应收票据及应收账款', '存货',
            '固定资产净额', '短期借款', '长期借款',
            '所有者权益(或股东权益)合计', '资产总计', '负债合计', 'year', 'quarter'
        ]
        add_missing_columns(balance_data, keep_list)

        # 保留重要列并重命名
        balance_data = balance_data[keep_list]

        balance_data.rename(columns={
            '报告日': 'report_date',
            '货币资金': 'cash_and_equivalents',
            '应收票据及应收账款': 'accounts_receivable',
            '存货': 'inventory',
            '固定资产净额': 'net_fixed_assets',
            '短期借款': 'short_term_borrowings',
            '长期借款': 'long_term_borrowings',
            '所有者权益(或股东权益)合计': 'total_equity',
            '资产总计': 'total_assets',
            '负债合计': 'total_liabilities'
        }, inplace=True)

        return balance_data

    except Exception as e:
        print(f"获取资产负债表数据时发生错误：{e} stock_code：{stock_code}")
        return None


def get_stock_cash_flow_sheet_data(stock_code: str) -> pd.DataFrame:
    """
    获取指定股票代码的所有年份和季度的现金流量表数据，保留重要列并转换为英文列名。

    Args:
        stock_code (str): 股票代码，例如 '600000'。

    Returns:
        pd.DataFrame: 包含现金流量表数据的 DataFrame，如果获取失败则返回 None。
                      DataFrame 包含以下英文列名：
                          - report_date: 报告日期
                          - net_cash_from_operating: 经营活动产生的现金流量净额
                          - net_cash_from_investing: 投资活动产生的现金流量净额
                          - net_cash_from_financing: 筹资活动产生的现金流量净额
                          - net_increase_in_cce: 现金及现金等价物净增加额
                          - year: 年份
                          - quarter: 季度
    """

    try:
        cash_flow_data = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")

        # 检查是否成功获取数据
        if cash_flow_data is None or cash_flow_data.empty:
            print(f"未能获取股票 {stock_code} 的现金流量表数据，请检查股票代码是否正确。")
            return None

        # 提取年份和季度信息
        cash_flow_data['year'] = cash_flow_data['报告日'].str[:4].astype(int)
        cash_flow_data['quarter'] = cash_flow_data['报告日'].str[4:6].astype(int) // 3

        keep_list = [
            '报告日', '经营活动产生的现金流量净额', '投资活动产生的现金流量净额',
            '筹资活动产生的现金流量净额', '现金及现金等价物净增加额',
            '期末现金及现金等价物余额', 'year', 'quarter'
        ]

        add_missing_columns(cash_flow_data, keep_list)
        # 保留重要列并重命名
        cash_flow_data = cash_flow_data[keep_list]

        cash_flow_data.rename(columns={
            '报告日': 'report_date',
            '经营活动产生的现金流量净额': 'net_cash_from_operating',
            '投资活动产生的现金流量净额': 'net_cash_from_investing',
            '筹资活动产生的现金流量净额': 'net_cash_from_financing',
            '现金及现金等价物净增加额': 'net_increase_in_cce',
            '期末现金及现金等价物余额': 'end_cash_and_cash_equivalents'
        }, inplace=True)

        return cash_flow_data

    except Exception as e:
        print(f"获取现金流量表数据时发生错误：{e} stock_code：{stock_code}")
        return None


