
from data.data_merging.merge_data import get_stock_v1_training_data
from data.data_merging.merge_data_v2 import get_stock_v2_training_data
from db import get_db_session
from db.predict_report import get_predict_report_by_date, bulk_insert_predict_report
from db.sh_index_daily import get_last_index_daily_date
from db.stock_list import get_predict_stock_list_data

from models.LSTMTransformer.predict import ModelPredictor

import datetime
import pandas as pd

from src.crawl.sync_daily_all import sync_daily_all
from src.training.parameter import get_config
from upload2aws.upload_to_dynamodb import import_2_aws_process


def predict_stock_list(stock_list: list, date_object: datetime.datetime = None,
                       version="simple_lstm_v1_2") -> pd.DataFrame:
    predictor = ModelPredictor(version)
    config = get_config(version)

    df = pd.DataFrame(columns=['report_date', 'stock_code', 'change_1d', 'change_3d', 'change_5d', 'change_10d'])
    if date_object is None:
        with get_db_session() as db:
            date_object = get_last_index_daily_date(db)

    # check if the report has been generated
    with get_db_session() as db:
        report_df = get_predict_report_by_date(db, date_object.strftime("%Y%m%d"))
    if not report_df.empty:
        return df

    for stock_code in stock_list:
        result = predict_stock(stock_code, predictor, date_object, config.data)
        if result is None:
            continue
        predict_data = {
            'report_date': date_object,
            'stock_code': stock_code,
            'change_1d': result[0],
            'change_3d': result[1],
            'change_5d': result[2],
            'change_10d': result[3],
            'model_version': version
        }
        df = pd.concat([df, pd.DataFrame.from_dict([predict_data])], ignore_index=True)

    return df


def predict_stock(stock_code: str, predictor: ModelPredictor, date: datetime.date, data_version= "v1"):
    if date is None:
        return None
    end_day = date
    # 计算100天前日期
    delta = datetime.timedelta(days=200)
    start_day = end_day - delta

    end_date = end_day.strftime("%Y%m%d")
    start_date = start_day.strftime("%Y%m%d")
    if data_version == "v2":
        stock_list = get_stock_v2_training_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
    else:
        stock_list = get_stock_v1_training_data(stock_code=stock_code, start_date=start_date, end_date=end_date)

    if stock_list is None or stock_list.empty or len(stock_list) <= 60:
        print("获取股票数据出错: " + stock_code)
        return None

    predict_data = stock_list.tail(n=60)
    predictions = predictor.predict(predict_data)

    return predictions[0]


def process_predict(report_date=None, sync_all=True, import_2_aws=True, version="simple_lstm_v1_2"):
    if report_date is None:
        with get_db_session() as db:
            report_date_object = get_last_index_daily_date(db)
    else:
        report_date_object = datetime.datetime.strptime(report_date, "%Y-%m-%d")

    if sync_all:
        sync_daily_all()
    with get_db_session() as db:
        stock_list = get_predict_stock_list_data(db)
    stock_code_list = stock_list["code"].values
    df = predict_stock_list(stock_code_list, report_date_object, version)
    with get_db_session() as db:
        bulk_insert_predict_report(db, df)
    if import_2_aws:
        with get_db_session() as db:
            df = get_predict_report_by_date(db, report_date)
        import_2_aws_process(report=df)
