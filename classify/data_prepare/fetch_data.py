from datetime import date, timedelta

import pandas as pd

from classify.get_classify_data import get_classify_xy_data_from_df
from data.data_merging.merge_data import get_stock_all_data
from db import get_db_session
from db.stock_list import get_all_stock_list_data


def get_training_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % 3 == 0]  # Select multiples of 3
    result = pd.concat([result, df[df.index % 3 == 1]])
    return result


def get_predict_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % 3 == 2]
    return result


def get_XY_all_list():
    df = get_training_list()
    all_x =[]
    all_y = []
    for index, row in df.iterrows():
        x_list, y_list = get_xy_data_by_stock_code(row["code"])
        if x_list is None or y_list is None:
            continue
        all_x = all_x + x_list
        all_y = all_y + y_list

    return all_x, all_y


def get_xy_data_by_stock_code(stock_code: str):
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    result = get_stock_all_data(stock_code=stock_code, start_date=five_years_ago.strftime("%Y%m%d"),
                                end_date=today.strftime("%Y%m%d"))

    x_list=[]
    y_list =[]
    id = 0
    if result is None:
        return None, None
    while id < len(result) - 70:
        target_df = result[id: id+70]
        x, y = get_classify_xy_data_from_df(target_df, [i for i in range(77)], "stock_close")
        x_list.append(x)
        y_list.append(y)
        id = id + 1
    return x_list, y_list

