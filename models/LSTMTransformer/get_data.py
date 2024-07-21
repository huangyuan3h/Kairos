import pandas as pd

DF_length = 70

X_length = 60


def get_xy_data_from_df(df: pd.DataFrame, feature_columns: list, target_column: str):
    if len(df) != 70:
        return None

    x = df.iloc[:, feature_columns]
    y_data = df.tail(10)[target_column].values
    x = x.head(X_length)

    current_close = x.tail(1)[target_column].values[0]
    y = [(y_data[i - 1] - current_close) * 100 / current_close for i in [1, 3, 5, 10]]

    return x, y
