from sklearn.preprocessing import StandardScaler

from data.data_merging import get_random_valid_data, get_stock_total_data, drop_columns_and_reset_index

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel

from models.LSTMTransformer.load_model import load_model
from models.LSTMTransformer.predict import predict

from src.training.parameter import get_model_params, get_training_params, get_data_params

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_predictions(expected, predictions):
    """
    评估预测值和预期值之间的差异。

    Args:
        expected (list): 预期值列表。
        predictions (list): 预测值列表。

    Returns:
        dict: 包含MSE、RMSE和MAE的字典。
    """
    mse = mean_squared_error(expected, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(expected, predictions)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

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
    # 评估预测值和预期值之间的差异
    evaluation_results = evaluate_predictions(expected, predictions)
    print("评估结果：", evaluation_results)


def predict_stock_list(stock_list: list):
    # 获取模型参数
    input_dim, hidden_dim, num_layers, num_heads, target_days = get_model_params()

    model = LSTMTransformerModel(input_dim, hidden_dim, num_layers, num_heads)
    # 获取训练参数
    batch_size, learning_rate, num_epochs, model_save_path = get_training_params()

    model = load_model(model, model_save_path)

    # 获取数据参数
    feature_columns, target_column = get_data_params()

    result = []

    import datetime

    # 获取当前时间
    now_time = datetime.datetime.now()

    # 计算 200 天前的时间
    delta = datetime.timedelta(days=200)
    before_200_days = now_time - delta
    scaler = StandardScaler()

    start_date = before_200_days.strftime("%Y%m%d")
    for code in stock_list:
        stock_list_200_day = get_stock_total_data(stock_code=code, start_date=start_date, n_days=200)
        predict_data = drop_columns_and_reset_index(stock_list_200_day[-60:])
        scaler.fit(predict_data)
        predictions = predict(model, predict_data, scaler, feature_columns)
        result.append(predictions)

    return result


