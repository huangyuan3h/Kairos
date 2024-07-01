from data.raw import get_currency_exchange_rates
from db import create_table
from import_2_db import import_exchange_rates

start_date = '20190101'

def main():
    # import_all_stocks_daily()
    create_table()
    import_exchange_rates()





if __name__ == "__main__":
    main()
