from data.data_preprocessing import merge_financial_data
from data.raw import get_stock_profit_sheet_data, \
    get_stock_balance_sheet_data, get_stock_cash_flow_sheet_data
from db import create_table
from import_2_db import import_exchange_rates
from import_2_db.import_sh_index_daily import import_sh_index_daily
from import_2_db.import_sz_index_daily import import_sz_index_daily

start_date = '20190101'


def main():
    # create_table()
    # import_sz_index_daily()
    code = '002594'
    profit = get_stock_profit_sheet_data(code)
    balance = get_stock_balance_sheet_data(code)
    cash_flow = get_stock_cash_flow_sheet_data(code)

    merged = merge_financial_data(profit, balance, cash_flow, code)
    print(merged)


if __name__ == "__main__":
    main()
