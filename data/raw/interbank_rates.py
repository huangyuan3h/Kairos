import akshare as ak
import pandas as pd

def get_shibor_rate() -> pd.DataFrame:
    """
    获取一段时间内的银行间同业拆借利率（Shibor）。

    Args:
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'。
        end_date (str): 结束日期，格式为 'YYYYMMDD'，例如 '20230101'。

    Returns:
        pd.DataFrame: 包含 Shibor 利率数据的 DataFrame，如果获取失败则返回 None。
                  DataFrame 包含以下列：
                      - date: 报告日期
                      - rate: 利率
                      - change: 涨跌
    """

    try:
        shibor_df = ak.rate_interbank(market="中国银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜")
        shibor_df.rename(columns={'报告日': 'date', '利率': 'rate', '涨跌': 'change'}, inplace=True)
        return shibor_df
    except Exception as e:
        print(f"获取银行间同业拆借利率数据时发生错误：{e}")
        return None