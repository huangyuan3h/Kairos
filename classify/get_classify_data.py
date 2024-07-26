import pandas as pd

DF_length = 70

X_length = 60


def get_classify_xy_data_from_df(df: pd.DataFrame, feature_columns: list, target_column: str):
    if len(df) != 70:
        return None

    x = df.iloc[:, feature_columns]
    y_data = df.tail(10)[target_column].values
    x = x.head(X_length)

    current_close = x.tail(1)[target_column].values[0]
    three_day_change = (y_data[0] - current_close) / current_close
    five_day_change = (y_data[2] - current_close) / current_close

    # 根据涨跌幅计算分类标签
    if three_day_change >= 0.04 and five_day_change >= 0.06:
        y = [2]  # 暴涨
    elif five_day_change <= -0.06:
        y = [0]  # 暴跌
    else:
        y = [1]  # 其他

    return x, y
