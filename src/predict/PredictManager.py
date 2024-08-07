

import pandas as pd

from days.days_predict import DaysPredictor
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from trend.trend_predict import TrendPredictor


class PredictManager:
    def __init__(self):
        self.feature_scaler = FeatureStandardScaler(data_version="v2")
        self.predictor_day1 = DaysPredictor(version="lstmTransformer", days=1)
        self.predictor_day2 = DaysPredictor(version="lstmTransformer", days=2)
        self.predictor_day3 = DaysPredictor(version="lstmTransformer", days=3)
        self.trend_predictor = TrendPredictor(version="lstmTransformer")

    def predict_all(self, stock_data: pd.DataFrame):
        if stock_data is None or stock_data.empty or len(stock_data) != 60:
            print("stock_data is empty or error ")
            return None

        p1 = self.predictor_day1.predict(stock_data)
        p2 = self.predictor_day2.predict(stock_data)
        p3 = self.predictor_day3.predict(stock_data)
        t = self.trend_predictor.predict(stock_data)

        return [p1[0][0], p2[0][0], p3[0][0], t[0][0]]
