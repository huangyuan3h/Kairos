import akshare as ak
import pandas as pd


def get_currency_exchange_rates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取一段时间内美元和欧元对人民币的汇率。

    Args:
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'。
        end_date (str): 结束日期，格式为 'YYYYMMDD'，例如 '20230101'。

    Returns:
        pd.DataFrame: 包含美元和欧元汇率数据的 DataFrame，如果获取失败则返回 None。
                  DataFrame 包含以下列：
                      - date: 日期
                      - USD_CNY: 美元对人民币汇率 (中行钞买价)
                      - EUR_CNY: 欧元对人民币汇率 (中行钞买价)
    """

    try:

        # 获取美元汇率
        usd_rates = ak.currency_boc_sina(symbol="美元", start_date=start_date, end_date=end_date)
        usd_rates = usd_rates[['日期', '中行钞买价']]
        usd_rates.rename(columns={'日期': 'date', '中行钞买价': 'USD_CNY'}, inplace=True)

        # 获取欧元汇率
        eur_rates = ak.currency_boc_sina(symbol="欧元", start_date=start_date, end_date=end_date)
        eur_rates = eur_rates[['日期', '中行钞买价']]
        eur_rates.rename(columns={'日期': 'date', '中行钞买价': 'EUR_CNY'}, inplace=True)

        # 合并数据
        merged_rates = pd.merge(usd_rates, eur_rates, on='date')

        return merged_rates

    except Exception as e:
        print(f"获取汇率数据时发生错误：{e}")
        return None


