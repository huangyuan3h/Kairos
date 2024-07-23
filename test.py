from models.LSTMTransformer.RandomStockData import RandomStockData
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler
from src.training.parameter import get_config



if __name__ == "__main__":

    config = get_config("simple_lstm_v2_1")
    # 获取模型参数
    mp = config.model_params
    tp = config.training_params
    dp = config.data_params
    Model = config.Model
    data_version = config.data

    feature_scaler = FeatureStandardScaler(data_version=data_version)
    target_scaler = TargetStandardScaler(data_version=data_version)
    generator = RandomStockData(dp.feature_columns, dp.target_column, feature_scaler, target_scaler, data_version)
    x, y = generator.get_data()
    print(x, y)
