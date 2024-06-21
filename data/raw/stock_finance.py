import akshare as ak
import pandas as pd

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
                          - net_profit_atsopc: 归属于母公司所有者的净利润
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

        # 保留重要列并重命名
        profit_data = profit_data[['报告日', '营业收入', '营业利润', '净利润', '归属于母公司的净利润', '基本每股收益', 'year', 'quarter']]
        profit_data.rename(columns={
            '报告日': 'report_date',
            '营业收入': 'revenue',
            '营业利润': 'operating_profit',
            '净利润': 'net_profit',
            '归属于母公司的净利润': 'net_profit_atsopc',
            '基本每股收益': 'basic_eps'
        }, inplace=True)

        return profit_data

    except Exception as e:
        print(f"获取利润表数据时发生错误：{e}")
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
                          - total_equity_atsopc: 归属于母公司股东的权益
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

        # 计算股东权益合计
        balance_data['股东权益合计'] = balance_data['归属于母公司股东的权益'] + balance_data['少数股东权益']

        # 保留重要列并重命名
        balance_data = balance_data[['报告日', '资产总计', '负债合计', '股东权益合计', '归属于母公司股东的权益', 'year', 'quarter']]
        balance_data.rename(columns={
            '报告日': 'report_date',
            '资产总计': 'total_assets',
            '负债合计': 'total_liabilities',
            '股东权益合计': 'total_equity',
            '归属于母公司股东的权益': 'total_equity_atsopc'
        }, inplace=True)

        return balance_data

    except Exception as e:
        print(f"获取资产负债表数据时发生错误：{e}")
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

        # 保留重要列并重命名
        cash_flow_data = cash_flow_data[
            ['报告日', '经营活动产生的现金流量净额', '投资活动产生的现金流量净额', '筹资活动产生的现金流量净额', '现金及现金等价物净增加额',
             'year', 'quarter']]
        cash_flow_data.rename(columns={
            '报告日': 'report_date',
            '经营活动产生的现金流量净额': 'net_cash_from_operating',
            '投资活动产生的现金流量净额': 'net_cash_from_investing',
            '筹资活动产生的现金流量净额': 'net_cash_from_financing',
            '现金及现金等价物净增加额': 'net_increase_in_cce'
        }, inplace=True)

        return cash_flow_data

    except Exception as e:
        print(f"获取现金流量表数据时发生错误：{e}")
        return None
