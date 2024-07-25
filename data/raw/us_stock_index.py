import akshare as ak
import pandas as pd


def get_us_stock_index_data(symbol: str = ".IXIC") -> pd.DataFrame:
    """
    获取美股股票指数数据。

    Args:
        symbol: 指数代码，默认为 ".IXIC" (纳斯达克指数)，可选值：{".IXIC", ".DJI", ".INX", ".NDX"}。

    Returns:
        pd.DataFrame: 包含美股股票指数数据的 DataFrame，如果获取失败则返回 None。
                      DataFrame 包含以下列：
                          - date: 日期
                          - open: 开盘价
                          - high: 最高价
                          - low: 最低价
                          - close: 收盘价
                          - volume: 成交量
                          - amount: 成交额
    """

    try:
        index_us_stock_sina_df = ak.index_us_stock_sina(symbol=symbol)
        return index_us_stock_sina_df
    except Exception as e:
        print(f"获取美股股票指数数据时发生错误：{e}")
        return None