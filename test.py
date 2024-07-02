from data.raw import get_currency_exchange_rates, get_szse_component_index
from db import create_table
from import_2_db import import_exchange_rates
from import_2_db.import_sh_index_daily import import_sh_index_daily
from import_2_db.import_sz_index_daily import import_sz_index_daily

start_date = '20190101'


def main():
    create_table()
    import_sz_index_daily()


if __name__ == "__main__":
    main()
