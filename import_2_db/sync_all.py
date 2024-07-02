from import_2_db import import_all_stocks_daily, import_exchange_rates
from import_2_db.import_sh_index_daily import import_sh_index_daily
from import_2_db.import_sz_index_daily import import_sz_index_daily


def sync_all():
    import_all_stocks_daily()
    import_exchange_rates()
    import_sh_index_daily()
    import_sz_index_daily()