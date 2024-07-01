from data.raw import get_stock_data_since, get_sh_a_stock_list, get_sz_a_stock_list
from db.database import get_db_session, create_table
from db.stock_daily import bulk_insert_stock_daily_data, get_last_stock_data_date
from datetime import datetime, timedelta, date

stock_start_date = datetime(2019, 1, 1)

create_table()


def calculate_day_diff(start_date:datetime, end_date:datetime) -> int:
    return (end_date - start_date).days


def get_next_day(day: datetime) -> datetime:
    one_day = timedelta(days=1)
    next_day = day + one_day
    return datetime.combine(next_day, datetime.min.time())


def import_single_stock_by_code(code: str):
    cursor = stock_start_date
    with get_db_session() as db:
        last_date = get_last_stock_data_date(db, code)
        if last_date is not None:
            cursor = get_next_day(last_date)

    # 计算结束日期
    end_date_obj = datetime.now()
    end_date = end_date_obj.strftime('%Y%m%d')

    offset = calculate_day_diff(cursor, end_date_obj)
    stock_list = get_stock_data_since(code, cursor.strftime('%Y%m%d'), end_date)

    # handle all the error here
    if stock_list.empty or stock_list is None or len(stock_list) > offset:
        return

    with get_db_session() as db:
        bulk_insert_stock_daily_data(db, stock_list)


def import_sh_stocks_daily():
    stock_list = get_sh_a_stock_list()
    for index, row in stock_list.iterrows():
        import_single_stock_by_code(row["code"])
        print("sh"+str(row["index"]) +"stock code:"+row["code"] + " imported...")


def import_sz_stocks_daily():
    stock_list = get_sz_a_stock_list()
    for index, row in stock_list.iterrows():
        import_single_stock_by_code(row["code"])
        print("sz" + str(row["index"]) + "stock code:" + row["code"] + " imported...")


def import_all_stocks_daily():
    import_sh_stocks_daily()
    import_sz_stocks_daily()