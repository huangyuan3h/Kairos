import pandas as pd

DF_length = 70

X_length = 60


def get_xy_trend_data_from_df(df: pd.DataFrame, feature_columns: list, target_column: str):
    if len(df) != 70:
        return None

    x = df.iloc[:, feature_columns]
    y_data = df.tail(10)[target_column].values
    x = x.head(X_length)

    current_close = x.tail(1)[target_column].values[0]
    change_percentages = [(y_data[i] - current_close) * 100 / current_close for i in range(10)]
    # get day 4 to day 10 value
    weights = [0.2688, 0.2151, 0.1795, 0.1539, 0.1344, 0.1194, 0.1075]
    y = sum([change_percentages[i + 3] * weights[i] for i in range(len(weights))])
    return x, y
