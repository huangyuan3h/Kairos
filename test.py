from data.raw import get_currency_exchange_rates, get_szse_component_index
from db import create_table
from import_2_db import import_exchange_rates

start_date = '20190101'


def main():
    sz_list = get_szse_component_index('20230101','20240101')
    print(sz_list)


if __name__ == "__main__":
    main()
