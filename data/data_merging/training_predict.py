from datetime import datetime

import pandas as pd

from data.data_merging.merge_data import year, get_random_available_date, get_n_year_later, get_random_code_from_df

from data.data_merging.merge_data_v2 import get_stock_v2_training_data
from db import get_db_session
from db.stock_list import get_all_stock_list_data


def get_training_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % 3 == 0]
    result = pd.concat([result, df[df.index % 3 == 1]])
    return result


def get_predict_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % 3 == 2]
    return result


def get_random_training_code() -> str:
    stock_list = get_training_list()
    return get_random_code_from_df(stock_list)


def get_random_predict_code() -> str:
    stock_list = get_predict_list()
    return get_random_code_from_df(stock_list)


def get_random_v2_training_data() -> pd.DataFrame:
    result = None

    while result is None or len(result) <= 200 * year:
        code = get_random_training_code()
        start_date = get_random_available_date()
        end_date = get_n_year_later(datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_v2_training_data(stock_code=code, start_date=start_date,
                                            end_date=end_date.strftime("%Y%m%d"))
    return result


def get_random_v2_predict_data() -> pd.DataFrame:
    result = None

    while result is None or len(result) <= 200 * year:
        code = get_random_predict_code()
        start_date = get_random_available_date()
        end_date = get_n_year_later(datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_v2_training_data(stock_code=code, start_date=start_date,
                                            end_date=end_date.strftime("%Y%m%d"))
    return result
