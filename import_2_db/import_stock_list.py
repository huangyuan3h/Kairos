from data.raw import get_sh_a_stock_list, get_sz_a_stock_list
from db import get_db_session
from db.stock_list import bulk_insert_stock_daily_data


def import_all_stock_list_data():
    stock_list = get_sh_a_stock_list()
    with get_db_session() as db:
        bulk_insert_stock_daily_data(db, stock_list)
    stock_list = get_sz_a_stock_list()
    with get_db_session() as db:
        bulk_insert_stock_daily_data(db, stock_list)