from data.data_preprocessing import merge_financial_data
from data.raw import get_stock_profit_sheet_data, get_stock_balance_sheet_data, get_stock_cash_flow_sheet_data, \
    get_sh_a_stock_list, get_sz_a_stock_list
from db import get_db_session
from db.stock_financial_data import bulk_insert_financial_data, get_last_index_daily_date
import datetime

import time

from db.stock_list import get_all_stock_list_data


def import_single_financial_by_code(code: str)-> bool:
    with get_db_session() as db:
        last_date = get_last_index_daily_date(db, code)

    today = datetime.date.today()
    three_months_ago = today - datetime.timedelta(days=90)
    if last_date is None or last_date > datetime.datetime.combine(three_months_ago, datetime.datetime.min.time()):
        return False

    profit = get_stock_profit_sheet_data(code)
    balance = get_stock_balance_sheet_data(code)
    cash_flow = get_stock_cash_flow_sheet_data(code)

    merged = merge_financial_data(profit, balance, cash_flow, code)

    if last_date is not None:
        merged = merged.loc[merged['report_date'] > last_date]

    with get_db_session() as db:
        bulk_insert_financial_data(db, merged)
    return True


def import_all_financial_report():
    with get_db_session() as db:
        stock_list = get_all_stock_list_data(db)
    counter = 0
    for index, row in stock_list.iterrows():
        result = import_single_financial_by_code(row["code"])
        if result:
            print("stock financial code:" + row["code"] + " imported...")
            counter = counter + 1
        if (counter + 1) % 10 == 0:
            time.sleep(60)
            counter = 0
