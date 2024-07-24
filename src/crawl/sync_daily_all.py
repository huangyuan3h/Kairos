from import_2_db import import_all_stocks_daily, import_exchange_rates, import_sh_index_daily, import_sz_index_daily, \
    import_shibor_rate
from import_2_db.import_vix_daily import import_etf_qvix


def sync_daily_all():
    import_all_stocks_daily()
    import_exchange_rates()
    import_sh_index_daily()
    import_sz_index_daily()
    import_shibor_rate()
    import_etf_qvix()