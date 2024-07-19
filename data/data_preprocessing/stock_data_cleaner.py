import pandas as pd


def clean_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    清洗股票数据，处理缺失值、异常值和进行特征工程。

    Args:
        stock_data (pd.DataFrame): 原始的股票数据，包含以下列：
                                    - date: 日期
                                    - code: 股票代码
                                    - open: 开盘价
                                    - close: 收盘价
                                    - high: 最高价
                                    - low: 最低价
                                    - volume: 成交量
                                    - amount: 成交额
                                    - ... (其他列)

    Returns:
        pd.DataFrame: 清洗后的股票数据，包含原始列和新增的特征列。
        :param for_training: 假如是训练模型的数据，删除不准的那些数据
    """

    # 1. 处理缺失值：使用前一日数据填充, 并删除首日nan
    stock_data.ffill(inplace=True)
    stock_data.dropna(inplace=True)

    # 2. 数据标准化：使用Z-score标准化方法
    # for col in ['stock_open', 'stock_high', 'stock_low', 'stock_volume', 'stock_amount']:
    #     stock_data[col] = zscore_standardization(stock_data[col])

    # 3. 特征工程：添加一些技术指标
    # 计算每日涨跌幅
    stock_data['daily_return'] = stock_data['stock_close'].pct_change()
    # 计算5日均线和20日均线, 设置 min_periods=1
    stock_data['ma5'] = stock_data['stock_close'].rolling(window=5, min_periods=1).mean()
    stock_data['ma20'] = stock_data['stock_close'].rolling(window=20, min_periods=1).mean()

    # 4. 统一日期格式
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data['date'] = stock_data['date'].dt.strftime('%Y%m%d')

    return stock_data





# list = get_stock_data_since('600000', '20230101', 100)
#
# cleaned_list = clean_stock_data(list)
#
# print(cleaned_list)
