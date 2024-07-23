import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.data_merging.merge_data import get_random_v1_data
from models.LSTMTransformer.get_data import get_xy_data_from_df
from models.LSTMTransformer.predict import ModelPredictor
from src.training.parameter import get_config


def evaluate_model(model_name: str, get_data_func, data_version = "v1") -> dict:
    """
    使用提供的数据获取函数评估模型的准确度。

    Args:
        model_name (str): 模型名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        dict: 包含评估指标的字典。
    """
    predictor = ModelPredictor(model_name)
    X, y_true = get_data_func(data_version=data_version)

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
        config = get_config(model_name)
        data_version = config.data
        results[model_name] = evaluate_model(model_name, get_data_func, data_version)
    return pd.DataFrame.from_dict(results, orient='index')


x_list = []
y_list = []

x_list_v2 = []
y_list_v2 = []

eval_data_list = []

batch_size = 1000


def get_my_data(data_version="v1"):
    config = get_config(data_version)
    # 获取模型参数
    dp = config.data_params

    if len(eval_data_list) == 0:
        for i in range(batch_size):

            random_data = get_random_v1_data()
            eval_data = random_data.tail(70)
            eval_data_list.append(eval_data)

            x, y = get_xy_data_from_df(eval_data, dp.feature_columns, dp.target_column)
            x_list.append(x)
            y_list.append(y)

    if data_version == "v1":
        return x_list, y_list
    else:
        return x_list_v2, y_list_v2
