import pandas as pd

DF_length = 70
X_length = 60


def get_classify_xy_data_from_df(df: pd.DataFrame, feature_columns: list, target_column: str):
    if len(df) != DF_length:
        return None

    x = df.iloc[:, feature_columns]
    y_data = df.tail(10)[target_column].values
    x = x.head(X_length)

    current_close = x.tail(1)[target_column].values[0]
    three_day_change = (y_data[0] - current_close) / current_close
    five_day_change = (y_data[2] - current_close) / current_close

    # 计算股票历史波动率 (例如过去 20 天的标准差)
    volatility = df[target_column].rolling(window=20).std().iloc[-1]

    # 根据波动率动态调整阈值
    up_threshold = 0.04 + 0.5 * volatility
    down_threshold = -0.04 - 0.5 * volatility

    if three_day_change >= up_threshold and five_day_change >= up_threshold * 1.5:
        y = [1]  # 趋势涨
    elif three_day_change <= down_threshold and five_day_change <= down_threshold * 1.5:
        y = [1]  # 趋势跌
    else:
        y = [0]  # 其他

    return x, y
