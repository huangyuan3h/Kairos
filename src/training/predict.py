from data.data_merging import get_stock_total_data

from models.LSTMTransformer.predict import ModelPredictor

import datetime


def predict_stock_list(stock_list: list):
    predictor = ModelPredictor()
    results = []
    for stock_code in stock_list:
        result = predict_stock(stock_code, predictor)
        results.append(result)
    return results


def predict_stock(stock_code: str, predictor: ModelPredictor):
    end_day = datetime.date.today()

    # 计算100天前日期
    delta = datetime.timedelta(days=100)
    start_day = end_day - delta

    end_date = end_day.strftime("%Y%m%d")
    start_date = start_day.strftime("%Y%m%d")

    stock_list = get_stock_total_data(stock_code=stock_code, start_date=start_date, end_date=end_date)

    if stock_list is None or stock_list.empty or len(stock_list) <= 60:
        print("获取股票数据出错！")
        return None

    predict_data = stock_list.tail(n=60)
    # 初始化预测器
    predictions = predictor.predict(predict_data)
    return predictions
