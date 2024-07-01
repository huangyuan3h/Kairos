import akshare as ak
import pandas as pd


def get_stock_data_since(stock_code: str, start_date: str, end_date: str, adjust: str = 'hfq') -> pd.DataFrame:
    """
    获取指定股票代码从某日起往后的n条记录

    Args:
        stock_code (str): 股票代码，例如 '600000'
        start_date (str): 开始日期，格式为 'YYYYMMDD'，例如 '20230101'
        end_date (str): 结束日期日期，格式为 'YYYYMMDD'，例如 '20230101'
        adjust (str, optional): 复权方式，默认为 'hfq' (后复权)，可选 'qfq' (前复权) 或 None (不复权)

    Returns:
        pd.DataFrame: 包含股票数据的 DataFrame，如果获取失败则返回 None
    """

    try:
        # 使用 akshare 获取股票数据
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date,
                                        adjust=adjust)

        # 检查是否成功获取数据
        if stock_data is None or stock_data.empty:
            print(f"未能获取股票 {stock_code} 的数据，请检查股票代码和日期是否正确。")
            return None

        stock_data.rename(columns={
            '日期': 'date',
            '股票代码': 'stock_code',
            '开盘': 'stock_open',
            '收盘': 'stock_close',
            '最高': 'stock_high',
            '最低': 'stock_low',
            '成交量': 'stock_volume',
            '成交额': 'stock_amount',
            '振幅': 'stock_amplitude',
            '涨跌幅': 'stock_change_percent',
            '涨跌额': 'stock_change',
            '换手率': 'stock_turnover_rate'
        }, inplace=True)
        return stock_data

    except Exception as e:
        print(f"获取股票数据时发生错误：{e}")
        return None


# list = get_stock_data_since('600000', '20230101', 100)
# print(list)