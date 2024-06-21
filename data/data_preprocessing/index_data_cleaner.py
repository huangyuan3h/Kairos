import pandas as pd

from data.raw import get_sse_composite_index
pd.options.mode.chained_assignment = None  # default='warn'

def clean_index_data(index_data: pd.DataFrame) -> pd.DataFrame:
    """
    清洗上证指数或深证成指数据。

    Args:
        index_data (pd.DataFrame): 包含指数数据的 DataFrame，列名需包含：
                                    - date: 日期
                                    - open: 开盘价
                                    - close: 收盘价
                                    - high: 最高价
                                    - low: 最低价
                                    - volume: 成交量
                                    - amount: 成交额

    Returns:
        pd.DataFrame: 清洗后的指数数据 DataFrame，包含以下列：
                      - date: 日期
                      - open: 开盘价
                      - close: 收盘价
                      - high: 最高价
                      - low: 最低价
                      - volume: 成交量 (经过异常值处理)
                      - amount: 成交额 (经过异常值处理)
    """

    # 检查数据是否为空
    if index_data is None or index_data.empty:
        print("指数数据为空，无法进行清洗。")
        return None

    # 检查必要的列是否存在
    required_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
    for col in required_columns:
        if col not in index_data.columns:
            raise ValueError(f"指数数据缺少必要的列：{col}")

    # 转换日期列为 datetime 类型
    index_data['date'] = pd.to_datetime(index_data['date'])
    index_data['date'] = index_data['date'].dt.strftime('%Y%m%d')

    # 处理成交量和成交额的异常值
    index_data['volume'] = clean_outliers(index_data['volume'])
    index_data['amount'] = clean_outliers(index_data['amount'])

    return index_data


def clean_outliers(data: pd.Series) -> pd.Series:
    """
    使用 IQR 方法处理数据中的异常值。

    Args:
        data (pd.Series): 待处理的数据 Series。

    Returns:
        pd.Series: 处理后的数据 Series。
    """

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[data < lower_bound] = lower_bound
    data[data > upper_bound] = upper_bound

    return data

# index_data = get_sse_composite_index(start_date='20230101', n_days=100)
#
# # 清洗数据
# cleaned_data = clean_index_data(index_data.copy())
#
# # 打印清洗后的数据
# print(cleaned_data.head())