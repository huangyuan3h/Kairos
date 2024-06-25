
import pandas as pd

from data.raw import get_stock_data_since


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
    """

    # 1. 处理缺失值：使用前一日数据填充
    stock_data.ffill().bfill()

    # 2. 处理异常值：使用 3σ 原则处理价格和成交量数据
    for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
        stock_data = remove_outliers_by_std(stock_data, col)

    # 3. 数据标准化：对价格和成交量数据进行 Min-Max 标准化
    for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
        stock_data[col] = min_max_scaling(stock_data[col])

    # 4. 特征工程：计算技术指标
    stock_data['ma5'] = stock_data['close'].rolling(window=5).mean()  # 5日移动平均线
    stock_data['ma10'] = stock_data['close'].rolling(window=10).mean()  # 10日移动平均线
    stock_data['rsi'] = calculate_rsi(stock_data['close'], 14)  # 计算 RSI 指标

    # 5. 统一日期格式
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data['date'] = stock_data['date'].dt.strftime('%Y%m%d')

    return stock_data


def remove_outliers_by_std(data: pd.DataFrame, column: str, std_threshold: float = 3.0) -> pd.DataFrame:
    """
    使用 3σ 原则移除指定列的异常值。

    Args:
        data (pd.DataFrame): 待处理的数据集。
        column (str): 需要处理的列名。
        std_threshold (float, optional): 标准差倍数阈值，默认为 3.0。

    Returns:
        pd.DataFrame: 处理后的数据集。
    """
    data_std = data[column].std()
    data_mean = data[column].mean()
    upper_limit = data_mean + std_threshold * data_std
    lower_limit = data_mean - std_threshold * data_std
    data = data[(data[column] < upper_limit) & (data[column] > lower_limit)]
    return data


def min_max_scaling(data: pd.Series) -> pd.Series:
    """
    对数据进行 Min-Max 标准化。

    Args:
        data (pd.Series): 待处理的数据。

    Returns:
        pd.Series: 标准化后的数据。
    """
    return (data - data.min()) / (data.max() - data.min())


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算相对强弱指数 (RSI)。

    Args:
        prices (pd.Series): 收盘价序列。
        period (int, optional): 计算 RSI 的周期，默认为 14。

    Returns:
        pd.Series: RSI 指标序列。
    """
    delta = prices.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = abs(loss.rolling(window=period).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi



# list = get_stock_data_since('600000', '20230101', 100)
#
# cleaned_list = clean_stock_data(list)
#
# print(cleaned_list)