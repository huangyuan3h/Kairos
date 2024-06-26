from sklearn.preprocessing import StandardScaler

from data.data_merging import get_random_valid_data

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel

from models.LSTMTransformer.load_model import load_model
from models.LSTMTransformer.predict import predict

from src.training.parameter import get_model_params, get_training_params, get_data_params


def main():
    # 获取模型参数
    input_dim, hidden_dim, num_layers, num_heads, target_days = get_model_params()

    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads)
    # 获取训练参数
    batch_size, learning_rate, num_epochs, model_save_path = get_training_params()

    model = load_model(model, model_save_path)

    # 获取数据参数
    feature_columns, target_column = get_data_params()

    data = get_random_valid_data()
    predict_data = data[0:60]
    scaler = StandardScaler()
    scaler.fit(predict_data)
    predictions = predict(model, predict_data, scaler, feature_columns)
    print("未来10天的预测数据：", predictions)


    expected_list = data['stock_change_percent'][60:70]
    expected = [expected_list[60], expected_list[:3].mean(), expected_list[:5].mean(), expected_list[:10].mean()]
    print(expected)


if __name__ == "__main__":
    main()
