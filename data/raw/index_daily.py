import akshare as ak
import pandas as pd


def get_sse_composite_index(start_date: str, n_days: int) -> pd.DataFrame:
    """
    获取上证指数在指定日期开始后的n个交易日的日线数据。

    Args:
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        n_days (int): 获取的天数

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

    return get_index_data('000001', start_date, n_days)


def get_szse_component_index(start_date: str, n_days: int) -> pd.DataFrame:
    """
    获取深证成指在指定日期开始后的n个交易日的日线数据。

    Args:
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        n_days (int): 获取的天数

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

    return get_index_data('399001', start_date, n_days)


def get_index_data(index_code: str, start_date: str, n_days: int) -> pd.DataFrame:
    """
    获取指定指数在指定日期开始后的n个交易日的日线数据。

    Args:
        index_code (str): 指数代码，例如 '000001.SH'
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        n_days (int): 获取的天数

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
        # 计算结束日期
        start_date_obj = pd.to_datetime(start_date, format='%Y%m%d')
        end_date_obj = start_date_obj + pd.Timedelta(days=n_days - 1)
        end_date = end_date_obj.strftime('%Y%m%d')
        index_data = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date=start_date, end_date=end_date)

        # 检查是否成功获取数据
        if index_data is None or index_data.empty:
            print(f"未能获取指数 {index_code} 的数据，请检查指数代码和日期是否正确。")
            return None

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

