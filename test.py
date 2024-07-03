from data.data_preprocessing import merge_financial_data
from data.raw import get_stock_profit_sheet_data, \
    get_stock_balance_sheet_data, get_stock_cash_flow_sheet_data
from db import create_table
from import_2_db import import_exchange_rates
from import_2_db.import_financial_data import import_single_financial_by_code, import_all_financial_daily
from import_2_db.import_sh_index_daily import import_sh_index_daily
from import_2_db.import_sz_index_daily import import_sz_index_daily

start_date = '20190101'


def main():
    create_table()
    # import_sz_index_daily()
    # code = '601456'
    #
    # import_single_financial_by_code(code)
    import_all_financial_daily()



if __name__ == "__main__":
    main()
