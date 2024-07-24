from data.data_merging.merge_data_v2 import get_random_data_all, get_random_v2_data
from data.raw.interbank_rates import get_shibor_rate
from db import create_table, get_db_session
from db.shibor_rates import get_shibor_rate_by_date_range
from import_2_db.import_shibor_rate import import_shibor_rate
import akshare as ak

from import_2_db.import_vix_daily import import_etf_qvix
from models.standardize.FeatureStandardScaler import FeatureStandardScaler

data = ['stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume', 'stock_amount',
        'stock_amplitude', 'stock_change_percent', 'stock_change', 'stock_turnover_rate', 'daily_return', 'ma5', 'ma20',
        'rsi', 'ATR', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist', 'VWAP', 'BOLL_mid',
        'BOLL_upper', 'BOLL_lower', 'day_of_week', 'month', 'quarter', 'is_end_of_week', 'is_end_of_month', 'Currency_USD_CNY',
        'Currency_EUR_CNY', 'Currency_USD_CNY_MA_5', 'Currency_USD_CNY_MA_20', 'Currency_EUR_CNY_MA_5', 'Currency_EUR_CNY_MA_20',
        'sse_open', 'sse_close', 'sse_high', 'sse_low', 'sse_change_percent',
        'sse_change', 'sse_turnover_rate', 'sse_daily_return', 'sse_ma5', 'sse_ma20', 'sse_rsi', 'szse_open', 'szse_close',
        'szse_high', 'szse_low', 'szse_change_percent', 'szse_change',
        'szse_turnover_rate', 'szse_daily_return', 'szse_ma5', 'szse_ma20', 'szse_rsi', 'rate_rate', 'rate_change',
        'qvix_open', 'qvix_high', 'qvix_low', 'qvix_close',  'gross_profit_margin', 'operating_profit_margin', 'net_profit_margin',
        'return_on_equity', 'return_on_assets', 'asset_turnover',
        'inventory_turnover', 'receivables_turnover',
        'current_ratio', 'quick_ratio', 'debt_to_asset_ratio', 'revenue_growth_rate', 'net_profit_growth_rate']

        # 'gross_profit_margin', 'operating_profit_margin', 'net_profit_margin',
        # 'return_on_equity', 'return_on_assets', 'asset_turnover',
        # 'inventory_turnover', 'receivables_turnover',
        # 'current_ratio', 'quick_ratio', 'debt_to_asset_ratio', 'revenue_growth_rate', 'net_profit_growth_rate']

        # 'sse_volume', 'sse_amount', 'sse_amplitude'
        # 'szse_volume', 'szse_amount', 'szse_amplitude'

if __name__ == "__main__":
    # feature_scaler = FeatureStandardScaler(data_version="v2")
    # list_data = get_random_v2_data()
    # return_data = feature_scaler.transform(list_data)
    print(data)

