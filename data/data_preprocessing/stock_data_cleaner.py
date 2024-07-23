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
    # 计算5日均线和20日均线
    stock_data['ma5'] = stock_data['stock_close'].rolling(window=5).mean()
    stock_data['ma20'] = stock_data['stock_close'].rolling(window=20).mean()
    # 计算RSI指标
    stock_data['rsi'] = calculate_rsi(stock_data['stock_close'])

    # 4. 统一日期格式
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data['date'] = stock_data['date'].dt.strftime('%Y%m%d')
    stock_data = add_technical_indicators(stock_data)
    stock_data = add_time_features(stock_data)
    stock_data = stock_data.drop(stock_data.head(20).index)
    return stock_data


def add_technical_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    添加更多技术指标。

    Args:
        stock_data (pd.DataFrame): 包含基本特征的股票数据。

    Returns:
        pd.DataFrame: 添加了更多技术指标的股票数据。
    """

    # 1. 波动性指标
    # 计算平均真实波动范围 (ATR)
    high_low = stock_data['stock_high'] - stock_data['stock_low']
    high_close = (stock_data['stock_high'] - stock_data['stock_close'].shift()).abs()
    low_close = (stock_data['stock_low'] - stock_data['stock_close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    stock_data['ATR'] = true_range.rolling(window=14).mean()

    # 2. 动量指标
    # 计算随机指标 (KDJ)
    low_min = stock_data['stock_low'].rolling(window=9).min()
    high_max = stock_data['stock_high'].rolling(window=9).max()
    stock_data['KDJ_K'] = ((stock_data['stock_close'] - low_min) / (high_max - low_min)) * 100
    stock_data['KDJ_D'] = stock_data['KDJ_K'].rolling(window=3).mean()
    stock_data['KDJ_J'] = 3 * stock_data['KDJ_D'] - 2 * stock_data['KDJ_K']

    # 计算移动平均收敛散度 (MACD)
    stock_data['EMA12'] = stock_data['stock_close'].ewm(span=12).mean()
    stock_data['EMA26'] = stock_data['stock_close'].ewm(span=26).mean()
    stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
    stock_data['MACD_signal'] = stock_data['MACD'].ewm(span=9).mean()
    stock_data['MACD_hist'] = stock_data['MACD'] - stock_data['MACD_signal']

    # 3. 其他指标
    # 计算成交量加权平均价格 (VWAP)
    stock_data['VWAP'] = (stock_data['stock_amount'] / stock_data['stock_volume']).cumsum() / stock_data[
        'stock_volume'].cumsum()

    # # 计算资金流量指标 (MFI)
    # typical_price = (stock_data['stock_high'] + stock_data['stock_low'] + stock_data['stock_close']) / 3
    # money_flow = typical_price * stock_data['stock_volume']
    # positive_flow = money_flow[money_flow > 0].rolling(window=14).sum()
    # negative_flow = money_flow[money_flow < 0].rolling(window=14).sum()
    # money_flow_ratio = positive_flow / negative_flow.abs()
    # stock_data['MFI'] = 100 - (100 / (1 + money_flow_ratio))

    return stock_data


def add_time_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    添加时间特征。

    Args:
        stock_data (pd.DataFrame): 包含日期特征的股票数据。

    Returns:
        pd.DataFrame: 添加了时间特征的股票数据。
    """
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    # 添加星期几特征
    stock_data['day_of_week'] = stock_data['date'].dt.dayofweek
    # 添加月份特征
    stock_data['month'] = stock_data['date'].dt.month
    # 添加季度特征
    stock_data['quarter'] = stock_data['date'].dt.quarter
    # 添加是否为交易日结束特征
    stock_data['is_end_of_week'] = stock_data['day_of_week'].isin([4, 5]).astype(int)
    # 添加是否为月末特征
    stock_data['is_end_of_month'] = stock_data['date'].dt.is_month_end.astype(int)
    stock_data['date'] = stock_data['date'].dt.strftime('%Y%m%d')
    return stock_data


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
