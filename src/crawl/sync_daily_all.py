from import_2_db import import_all_stocks_daily, import_exchange_rates, import_sh_index_daily, import_sz_index_daily


def sync_daily_all():
    import_all_stocks_daily()
    import_exchange_rates()
    import_sh_index_daily()
    import_sz_index_daily()