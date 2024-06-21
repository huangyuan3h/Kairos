import pandas as pd


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
    """

    # 1. 处理缺失值：使用前一日数据填充
    df.fillna(method='ffill', inplace=True)

    # 2. 处理异常值：汇率一般不会出现极端异常值，可以不做处理

    # 3. 数据类型转换：将日期列转换为 datetime 类型
    df['date'] = pd.to_datetime(df['date'])

    # 4. 特征工程：可以考虑计算汇率的移动平均线、波动率等指标

    return df

