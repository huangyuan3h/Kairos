import pandas as pd

# from data.raw import get_currency_exchange_rates


def clean_currency_exchange_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗汇率数据。

    Args:
        df (pd.DataFrame): 原始汇率数据，包含以下列：
                            - date: 日期
                            - USD_CNY: 美元对人民币汇率 (中行钞买价)
                            - EUR_CNY: 欧元对人民币汇率 (中行钞买价)

    Returns:
        pd.DataFrame: 清洗后的汇率数据，包含以下列：
                        - date: 日期
                        - USD_CNY: 美元对人民币汇率 (中行钞买价)
                        - EUR_CNY: 欧元对人民币汇率 (中行钞买价)
                        - USD_CNY_MA_5: 美元对人民币汇率5日移动平均线
                        - USD_CNY_MA_20: 美元对人民币汇率20日移动平均线
                        - EUR_CNY_MA_5: 欧元对人民币汇率5日移动平均线
                        - EUR_CNY_MA_20: 欧元对人民币汇率20日移动平均线
    """

    # 1. 处理缺失值：使用前一日数据填充
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # 2. 处理异常值：汇率一般不会出现极端异常值，可以不做处理

    # 3. 数据类型转换：将日期列转换为 datetime 类型
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.strftime('%Y%m%d')

    # 4. 特征工程：计算汇率的移动平均线、波动率等指标
    df['USD_CNY_MA_5'] = df['USD_CNY'].rolling(window=5).mean()  # 计算美元对人民币汇率 5 日移动平均线
    df['USD_CNY_MA_20'] = df['USD_CNY'].rolling(window=20).mean()  # 计算美元对人民币汇率 20 日移动平均线
    df['EUR_CNY_MA_5'] = df['EUR_CNY'].rolling(window=5).mean()  # 计算欧元对人民币汇率 5 日移动平均线
    df['EUR_CNY_MA_20'] = df['EUR_CNY'].rolling(window=20).mean()  # 计算欧元对人民币汇率 20 日移动平均线
    df = df.drop(df.head(20).index)
    return df

# list = get_currency_exchange_rates('20230101', 100)
#
# cleaned_list = clean_currency_exchange_rates(list)
# print(cleaned_list)