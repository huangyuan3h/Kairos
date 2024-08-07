import pandas as pd
import torch
from sklearn.metrics import r2_score

from data.data_merging.training_predict import get_random_v2_data_by_type
from sklearn.metrics import accuracy_score

from trend.get_trend_data import get_xy_trend_data_from_df
from trend.trend_parameter import get_trend_config
from trend.trend_predict import TrendPredictor


def evaluate_model(model_name: str, get_data_func) -> dict:
    """
    使用提供的数据获取函数评估模型的准确度，并考虑不同时间步的权重。

    Args:
        model_name (str): 模型名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        dict: 包含评估指标的字典。
    """
    predictor = TrendPredictor(model_name)
    X, y_true = get_data_func(model_name)
    # 使用模型进行预测
    predictions = []
    for x in X:
        prediction = predictor.predict(x)
        predictions.append(prediction[0])  # 获取预测值
    predictions = torch.stack([torch.tensor(p) for p in predictions])
    y_true = torch.tensor(y_true)

    # 计算加权指标
    mae = torch.abs(predictions - y_true).mean()
    mse = ((predictions - y_true) ** 2).mean()
    rmse = mse ** 0.5
    r2 = r2_score(y_true.detach().numpy(), predictions.detach().numpy())

    # 计算方向预测准确率
    direction_predictions = torch.where(predictions > 0, 1, 0)
    direction_true = torch.where(y_true > 0, 1, 0)
    accuracy = accuracy_score(direction_true.cpu().detach().numpy(), direction_predictions.cpu().detach().numpy())

    return {
        "MAE": mae.item(),
        "MSE": mse.item(),
        "RMSE": rmse.item(),
        "R2": r2,
        "Accuracy": accuracy,
    }


def compare_trend_models(model_1_name: str, model_2_name: str, get_data_func) -> pd.DataFrame:
    """
    使用提供的数据获取函数比较两个模型的性能。

    Args:
        model_1_name (str): 第一个模型的名称。
        model_2_name (str): 第二个模型的名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。
        days (number): 预测第几天
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


def get_trend_data(model_name="v1"):
    config = get_trend_config(model_name)
    # 获取模型参数
    dp = config.data_params
    if len(x_list) == 0:
        for i in range(batch_size):
            random_data = get_random_v2_data_by_type("test")
            eval_data = random_data.tail(70)
            x, y = get_xy_trend_data_from_df(eval_data, dp.feature_columns, dp.target_column)
            x_list.append(x)
            y_list.append(y)

    return x_list, y_list
