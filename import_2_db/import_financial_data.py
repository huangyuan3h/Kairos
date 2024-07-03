from data.data_preprocessing import merge_financial_data
from data.raw import get_stock_profit_sheet_data, get_stock_balance_sheet_data, get_stock_cash_flow_sheet_data
from db import get_db_session
from db.stock_financial_data import bulk_insert_financial_data, get_last_index_daily_date
import datetime


def import_single_financial_by_code(code: str):

    with get_db_session() as db:
        last_date = get_last_index_daily_date(db, code)

    today = datetime.date.today()
    three_months_ago = today - datetime.timedelta(days=90)
    if last_date is not None and last_date <= datetime.datetime.combine(three_months_ago, datetime.datetime.min.time()):
        return

    profit = get_stock_profit_sheet_data(code)
    balance = get_stock_balance_sheet_data(code)
    cash_flow = get_stock_cash_flow_sheet_data(code)

    merged = merge_financial_data(profit, balance, cash_flow, code)

    if last_date is not None:
        merged = merged.loc[merged['report_date'] > last_date]

    with get_db_session() as db:
        bulk_insert_financial_data(db, merged)
