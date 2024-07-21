from models.LSTMTransformer.RandomStockData import RandomStockData
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler
from src.training.parameter import get_config


def main():
    config = get_config("v1")
    # 获取模型参数
    mp = config.model_params
    tp = config.training_params
    dp = config.data_params
    Model = config.Model

    feature_scaler = FeatureStandardScaler()
    target_scaler = TargetStandardScaler()
    generator = RandomStockData(dp.feature_columns, dp.target_column, feature_scaler, target_scaler)
    data = generator.get_data()
    print(data)


if __name__ == "__main__":
    main()
