from models.LSTMTransformer.RandomStockData import RandomStockData
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler
from src.training.parameter import get_data_params


def main():
    feature_columns, target_column = get_data_params()
    feature_scaler = FeatureStandardScaler()
    target_scaler = TargetStandardScaler()
    generator = RandomStockData(feature_columns, target_column, feature_scaler, target_scaler)
    data = generator.get_data()
    print(data)


if __name__ == "__main__":
    main()
