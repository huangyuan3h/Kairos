import akshare as ak
import pandas as pd


def get_sse_composite_index(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取上证指数在指定日期开始后的n个交易日的日线数据。

    Args:
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        end_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'

    Returns:
        pd.DataFrame: 包含上证指数数据的 DataFrame，如果获取失败则返回 None
                      DataFrame 包含以下列（列名已转换为英文）：
                          - date: 日期
                          - open: 开盘价
                          - close: 收盘价
                          - high: 最高价
                          - low: 最低价
                          - volume: 成交量
                          - amount: 成交额
    """

    return get_index_data('000001', start_date, end_date)


def get_szse_component_index(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取深证成指在指定日期开始后的n个交易日的日线数据。

    Args:
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        end_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'

    Returns:
        pd.DataFrame: 包含深证成指数据的 DataFrame，如果获取失败则返回 None
                      DataFrame 包含以下列（列名已转换为英文）：
                          - date: 日期
                          - open: 开盘价
                          - close: 收盘价
                          - high: 最高价
                          - low: 最低价
                          - volume: 成交量
                          - amount: 成交额
    """

    return get_index_data('399001', start_date, end_date)


def get_index_data(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定指数在指定日期开始后的n个交易日的日线数据。

    Args:
        index_code (str): 指数代码，例如 '000001.SH'
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        end_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'

    Returns:
        pd.DataFrame: 包含指数数据的 DataFrame，如果获取失败则返回 None
                      DataFrame 包含以下列（列名已转换为英文）：
                          - date: 日期
                          - open: 开盘价
                          - close: 收盘价
                          - high: 最高价
                          - low: 最低价
                          - volume: 成交量
                          - amount: 成交额
    """

    try:
        index_data = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date=start_date, end_date=end_date)

        index_data.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change',
            '换手率': 'turnover_rate'
        }, inplace=True)

        return index_data

    except Exception as e:
        print(f"获取指数数据时发生错误：{e}")
        return None



# list = get_sse_composite_index('20230101', 100)
#
# print(list)