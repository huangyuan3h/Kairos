from datetime import datetime

import pandas as pd

from data.data_merging.merge_data import year, get_random_available_date, get_n_year_later, get_random_code_from_df

from data.data_merging.merge_data_v2 import get_stock_v2_training_data
from db import get_db_session
from db.stock_list import get_all_stock_list_data

DATA_CLASSIFY = 5


def get_training_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % DATA_CLASSIFY == 0]
    result = pd.concat([result, df[df.index % DATA_CLASSIFY == 1]])
    result = pd.concat([result, df[df.index % DATA_CLASSIFY == 2]])
    return result


def get_verify_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % DATA_CLASSIFY == 3]
    return result


def get_test_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % DATA_CLASSIFY == 4]
    return result


def get_random_training_code() -> str:
    stock_list = get_training_list()
    return get_random_code_from_df(stock_list)


def get_random_verify_code() -> str:
    stock_list = get_verify_list()
    return get_random_code_from_df(stock_list)


def get_random_test_code() -> str:
    stock_list = get_verify_list()
    return get_random_code_from_df(stock_list)


'''
training: 训练数据集
verify： 验证数据集，用来early stop
test： eval, 测试数据集
'''
code_function_mapping = {"training": get_random_training_code, "verify": get_random_verify_code,
                         "test": get_random_test_code}


def get_random_v2_data_by_type(code_type="training") -> pd.DataFrame:
    result = None

    code_function = code_function_mapping[code_type]

    while result is None or len(result) <= 200 * year:
        code = code_function()
        start_date = get_random_available_date()
        end_date = get_n_year_later(datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_v2_training_data(stock_code=code, start_date=start_date,
                                            end_date=end_date.strftime("%Y%m%d"))
    return result
