from data.data_merging import get_stock_total_data
from db import get_db_session
from db.predict_report import get_predict_report_by_date
from db.sh_index_daily import get_last_index_daily_date

from models.LSTMTransformer.predict import ModelPredictor

import datetime
import pandas as pd


def predict_stock_list(stock_list: list, date: datetime.date = None) -> pd.DataFrame:
    predictor = ModelPredictor()

    df = pd.DataFrame(columns=['report_date', 'stock_code', 'change_1d', 'change_3d', 'change_5d', 'change_10d'])
    if date is None:
        with get_db_session() as db:
            date = get_last_index_daily_date(db)

    # check if the report has been generated
    with get_db_session() as db:
        report_df = get_predict_report_by_date(db, date.strftime("%Y%m%d"))
    if not report_df.empty:
        return df

    for stock_code in stock_list:
        result = predict_stock(stock_code, predictor, date)
        if result is None:
            continue
        predict_data = {
            'report_date': date,
            'stock_code': stock_code,
            'change_1d': result[3],
            'change_3d': result[2],
            'change_5d': result[1],
            'change_10d': result[0],
        }
        df = pd.concat([df, pd.DataFrame.from_dict([predict_data])], ignore_index=True)

    return df


def predict_stock(stock_code: str, predictor: ModelPredictor, date: datetime.date):
    if date is None:
        return None
    end_day = date
    # 计算100天前日期
    delta = datetime.timedelta(days=200)
    start_day = end_day - delta

    end_date = end_day.strftime("%Y%m%d")
    start_date = start_day.strftime("%Y%m%d")

    stock_list = get_stock_total_data(stock_code=stock_code, start_date=start_date, end_date=end_date)

    if stock_list is None or stock_list.empty or len(stock_list) <= 60:
        print("获取股票数据出错: " + stock_code)
        return None

    predict_data = stock_list.head(60)
    predictions = predictor.predict(predict_data)

    return predictions[0]
