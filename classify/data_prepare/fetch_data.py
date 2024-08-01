from datetime import date, timedelta

import pandas as pd
import torch

from classify.RandomStockDataClassify import add_noise
from classify.get_classify_data import get_classify_xy_data_from_df

from data.data_merging.merge_data_v2 import  get_stock_v2_training_data
from db import get_db_session
from db.stock_list import get_all_stock_list_data
from models.standardize.FeatureStandardScaler import FeatureStandardScaler


def get_training_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % 20 == 0]
    result = pd.concat([result, df[df.index % 20 == 1]])
    return result


def get_predict_list():
    with get_db_session() as db:
        df = get_all_stock_list_data(db)
    result = df[df.index % 20 == 2]
    return result


def get_XY_all_list():
    df = get_training_list()
    all_x = []
    all_y = []
    feature_scaler = FeatureStandardScaler(data_version="v2")
    for index, row in df.iterrows():
        x_list, y_list = get_xy_data_by_stock_code(row["code"], feature_scaler)
        if x_list is None or y_list is None:
            continue
        all_x = all_x + x_list
        all_y = all_y + y_list
    return all_x, all_y


def get_xy_data_by_stock_code(stock_code: str, feature_scaler=None):
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    result = get_stock_v2_training_data(stock_code=stock_code, start_date=five_years_ago.strftime("%Y%m%d"),
                                        end_date=today.strftime("%Y%m%d"))

    x_list = []
    y_list = []
    id = 0
    if result is None:
        return None, None
    while id < len(result) - 70:
        target_df = result[id: id + 70]
        x, y = get_classify_xy_data_from_df(target_df, [i for i in range(77)], "stock_close")

        x = add_noise(x)
        x = torch.tensor(feature_scaler.transform(x))
        y = torch.tensor(y)

        x_list.append(x)
        y_list.append(y)
        id = id + 1
    return x_list, y_list
