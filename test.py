from data.data_merging.column_choose import filter_features_correlation, select_k_best_features
from data.data_merging.merge_data_v2 import get_random_total_data

column_to_keep = ['stock_close', 'stock_volume', 'stock_amplitude', 'stock_change_percent', 'rsi', 'ATR',
                  'KDJ_K',
                  'KDJ_D', 'KDJ_J', 'MACD', 'MACD_signal', 'MACD_hist', 'VWAP', 'month', 'day_of_week',
                  'is_end_of_week', 'is_end_of_month',
                  'Currency_EUR_CNY', 'Currency_EUR_CNY_MA_20', 'sse_open',
                  'sse_volume', 'sse_amplitude', 'sse_change_percent', 'sse_daily_return', 'sse_rsi', 'szse_volume',
                  'szse_amount', 'szse_amplitude', 'szse_change_percent', 'szse_daily_return', 'szse_rsi',
                  'cash_and_equivalents', 'accounts_receivable', 'inventory',
                  'net_fixed_assets', 'short_term_borrowings', 'long_term_borrowings', 'total_equity',
                  'net_cash_from_investing', 'net_cash_from_financing', 'net_increase_in_cce', 'receivables_turnover',
                  'current_ratio']


def main():
    # stock_list = get_random_total_data()
    # stock_list['stock_return'] = (stock_list['stock_close'].shift(-1) - stock_list['stock_close']) / stock_list[
    #     'stock_close']
    # stock_list = stock_list.ffill().bfill().fillna(value=0)
    # x = stock_list[column_to_keep]
    # y = stock_list['stock_return']
    #
    # df, columns = select_k_best_features(x, y, 20)
    # print(df, columns)

    print(len(column_to_keep))


if __name__ == "__main__":
    main()
