import akshare as ak
import pandas as pd


def get_50etf_qvix() -> pd.DataFrame:
    """
    获取 50ETF 期权波动率指数 (QVIX) 数据。

    Returns:
        pd.DataFrame: 包含 50ETF 期权波动率指数 (QVIX) 数据的 DataFrame，如果获取失败则返回 None。
                  DataFrame 包含以下列：
                      - date: 日期
                      - open: 开盘价
                      - high: 最高价
                      - low: 最低价
                      - close: 收盘价
    """

    try:
        qvix_df = ak.index_option_50etf_qvix()
        return qvix_df
    except Exception as e:
        print(f"获取 50ETF 期权波动率指数 (QVIX) 数据时发生错误：{e}")
        return None
