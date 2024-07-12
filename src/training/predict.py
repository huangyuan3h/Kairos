from data.data_merging import get_stock_total_data

from models.LSTMTransformer.predict import ModelPredictor

import datetime
import pandas as pd


def predict_stock_list(stock_list: list) -> pd.DataFrame:
    predictor = ModelPredictor()

    df = pd.DataFrame(columns=['report_date', 'stock_code', 'change_1d', 'change_3d', 'change_5d', 'change_10d'])
    for stock_code in stock_list:
        result = predict_stock(stock_code, predictor)
        if result is None :
            continue
        predict_data = {
            'report_date': datetime.date.today(),
            'stock_code': stock_code,
            'change_1d': result[0],
            'change_3d': result[1],
            'change_5d': result[2],
            'change_10d': result[3],
        }
        df = pd.concat([df, pd.DataFrame.from_dict([predict_data])], ignore_index=True)

    return df


def predict_stock(stock_code: str, predictor: ModelPredictor):
    end_day = datetime.date.today() + datetime.timedelta(days=1)
    # 计算100天前日期
    delta = datetime.timedelta(days=160)
    start_day = end_day - delta

    end_date = end_day.strftime("%Y%m%d")
    start_date = start_day.strftime("%Y%m%d")

    stock_list = get_stock_total_data(stock_code=stock_code, start_date=start_date, end_date=end_date)

    if stock_list is None or stock_list.empty or len(stock_list) <= 60:
        print("获取股票数据出错！")
        return None

    predict_data = stock_list.tail(n=60)
    predictions = predictor.predict(predict_data)

    return predictions[0]
