from datetime import datetime, timedelta

import pandas as pd

from data.data_merging.merge_data_v2 import get_stock_v2_training_data
from db import get_db_session
from db.predict_report import get_predict_report_by_date
from db.sh_index_daily import get_last_index_daily_date
from src.predict.PredictManager import PredictManager


def get_df_by_code_date(stock_code: str, date: datetime.date):
    if date is None:
        return None
    end_day = date
    # 计算100天前日期
    delta = timedelta(days=200)
    start_day = end_day - delta

    end_date = end_day.strftime("%Y%m%d")
    start_date = start_day.strftime("%Y%m%d")
    stock_list = get_stock_v2_training_data(stock_code=stock_code, start_date=start_date, end_date=end_date)

    if stock_list is None or stock_list.empty or len(stock_list) <= 60:
        print("获取股票数据出错: " + stock_code)
        return None

    data = stock_list.tail(n=60)
    return data


def predict_all(stock_list: list, date_object: datetime = None):
    df = pd.DataFrame(
        columns=['report_date', 'stock_code', 'change_1d', 'change_2d', 'change_3d', 'operation_1d', 'operation_2d',
                 'trend'])
    if date_object is None:
        with get_db_session() as db:
            date_object = get_last_index_daily_date(db)

    # check if the report has been generated
    with get_db_session() as db:
        report_df = get_predict_report_by_date(db, date_object.strftime("%Y%m%d"))
    if not report_df.empty:
        return df

    pm = PredictManager()
    for stock_code in stock_list:
        data = get_df_by_code_date(stock_code, date_object)
        result = pm.predict_all(data)
        if result is None:
            print("predict stock error, code:" + stock_code)
            continue
        predict_data = {
            'report_date': date_object,
            'stock_code': stock_code,
            'change_1d': result[0],
            'change_2d': result[1],
            'change_3d': result[2],
            'trend': result[3],
            'operation_1d': result[4],
            'operation_2d': result[5],
        }
        df = pd.concat([df, pd.DataFrame.from_dict([predict_data])], ignore_index=True)
    return df
