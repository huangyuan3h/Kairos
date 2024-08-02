import pandas as pd

DF_length = 70

X_length = 60


def get_xy_days_data_from_df(df: pd.DataFrame, feature_columns: list, target_column: str,days=1):
    if len(df) != 70:
        return None

    x = df.iloc[:, feature_columns]
    y_data = df.tail(10)[target_column].values
    x = x.head(X_length)

    current_close = x.tail(1)[target_column].values[0]
    change_percentages = [(y_data[i] - current_close) * 100 / current_close for i in range(10)]
    y = [change_percentages[days - 1]]
    return x, y
