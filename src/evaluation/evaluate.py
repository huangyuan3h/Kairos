import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.data_merging.merge_data import get_random_full_data
from models.LSTMTransformer.predict import ModelPredictor


def evaluate_model(model_name: str, get_data_func) -> dict:
    """
    使用提供的数据获取函数评估模型的准确度。

    Args:
        model_name (str): 模型名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        dict: 包含评估指标的字典。
    """
    predictor = ModelPredictor(model_name)
    X, y_true = get_data_func()

    # 使用模型进行预测
    predictions = []
    for x in X:
        prediction = predictor.predict(x)
        predictions.append(prediction[0])  # 获取预测值

    # 计算评估指标
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, predictions)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def compare_models(model_1_name: str, model_2_name: str, get_data_func) -> pd.DataFrame:
    """
    使用提供的数据获取函数比较两个模型的性能。

    Args:
        model_1_name (str): 第一个模型的名称。
        model_2_name (str): 第二个模型的名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        pd.DataFrame: 包含两个模型评估指标的 DataFrame。
    """
    results = {}
    for model_name in [model_1_name, model_2_name]:
        results[model_name] = evaluate_model(model_name, get_data_func)
    return pd.DataFrame.from_dict(results, orient='index')


x_list = []
y_list = []

batch_size = 1000


def get_my_data():
    if len(y_list) == 0:
        for i in range(batch_size):
            random_data = get_random_full_data()
            eval_data = random_data.tail(70)
            x = eval_data.head(60)
            y_data = eval_data.tail(10)["stock_close"].values
            current_close = x.tail(1)["stock_close"].values[0]
            y = [(y_data[i - 1] - current_close) * 100 / current_close for i in [1, 3, 5, 10]]
            x_list.append(x)
            y_list.append(y)

    return x_list, y_list





