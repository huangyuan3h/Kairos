import pandas as pd

DF_length = 70

X_length = 60

'''
trend > 0: 表示上升趋势，值越大，上升趋势越强。
trend < 0: 表示下降趋势，值越小，下降趋势越强。
trend ≈ 0: 表示横盘趋势，价格波动较小。
'''


def get_xy_operation_data_from_df(df: pd.DataFrame, feature_columns: list, days=1):
    if len(df) != 70:
        return None

    x = df.iloc[:, feature_columns]
    x = x.head(X_length)

    y_data = df.tail(10)

    buy_day = y_data.iloc[0]
    sell_day = y_data.iloc[days]

    buy_price = (buy_day["stock_high"] + buy_day["stock_low"]) / 2
    sell_price = (sell_day["stock_high"] + sell_day["stock_low"]) / 2

    y_operation = (sell_price - buy_price) * 100 / buy_price
    y = [y_operation]

    return x, y
