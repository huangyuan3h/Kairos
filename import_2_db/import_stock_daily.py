from data.raw import get_stock_data_since
from db.database import get_db_session
from db.stock_daily import bulk_insert_stock_daily_data, get_last_stock_data_date
from datetime import datetime

start_date = '20190101'


def get_next_day(day: datetime) -> datetime:
    one_day = datetime.timedelta(days=1)
    return day + one_day


def import_single_stock_by_code(code: str):
    cursor = start_date
    with get_db_session() as db:
        last_date = get_last_stock_data_date(db, code)
        if last_date is not None:
            cursor = get_next_day(last_date)

    is_empty = False
    while not is_empty:
        stock_list = get_stock_data_since(code, cursor, 300)
        if len(stock_list) == 0:
            is_empty = True

        with get_db_session() as db:
            bulk_insert_stock_daily_data(db, stock_list)

        last_date = stock_list['date'].iloc[-1]

        next_day = get_next_day(last_date)
        cursor = next_day

    print(stock_list['date'].iloc[-1])
