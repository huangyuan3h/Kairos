import pandas as pd

DF_length = 70

X_length = 60

'''
trend > 0: 表示上升趋势，值越大，上升趋势越强。
trend < 0: 表示下降趋势，值越小，下降趋势越强。
trend ≈ 0: 表示横盘趋势，价格波动较小。
'''


def get_xy_trend_data_from_df(df: pd.DataFrame, feature_columns: list, target_column: str):
    if len(df) != 70:
        return None

    x = df.iloc[:, feature_columns]
    x = x.head(X_length)

    closing_prices = df[target_column].tail(10).values
    # Calculate the price difference between day 10 and day 4
    price_difference = closing_prices[-1] - closing_prices[3]
    # Calculate the average closing price over the 7-day period
    average_price = closing_prices[3:].mean()
    # Normalize the price difference by the average price
    trend = price_difference / average_price

    y = [trend]
    return x, y
