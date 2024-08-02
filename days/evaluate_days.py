
import pandas as pd
import torch
from sklearn.metrics import r2_score

from data.data_merging.merge_data_v2 import get_random_data_all, keep_column_v2
from days.days_predict import DaysPredictor
from days.get_days_data import get_xy_days_data_from_df
from src.training.parameter import get_config


def evaluate_model(model_name: str, get_data_func, days=1) -> dict:
    """
    使用提供的数据获取函数评估模型的准确度，并考虑不同时间步的权重。

    Args:
        model_name (str): 模型名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        dict: 包含评估指标的字典。
    """
    predictor = DaysPredictor(model_name, days)
    X, y_true = get_data_func(model_name, days)
    # 使用模型进行预测
    predictions = []
    for x in X:
        prediction = predictor.predict(x)
        predictions.append(prediction[0])  # 获取预测值
    predictions = torch.stack([torch.tensor(p) for p in predictions])
    y_true = torch.tensor(y_true)


    # 计算加权指标
    weighted_mae = torch.abs(predictions - y_true).mean()
    weighted_mse = ((predictions - y_true) ** 2).mean()
    weighted_rmse = weighted_mse ** 0.5
    weighted_r2 = r2_score(y_true.view(-1), predictions.view(-1))



    return {
        "Weighted MAE": weighted_mae.item(),
        "Weighted MSE": weighted_mse.item(),
        "Weighted RMSE": weighted_rmse.item(),
        "Weighted R2": weighted_r2,
    }


def compare_days_models(model_1_name: str, model_2_name: str, get_data_func, days=1) -> pd.DataFrame:
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

        results[model_name] = evaluate_model(model_name, get_data_func, days)
    return pd.DataFrame.from_dict(results, orient='index')


eval_data_list = []

batch_size = 500


def get_days_data(model_name="v1", days = 1):
    x_list = []
    y_list = []

    config = get_config(model_name)
    # 获取模型参数
    dp = config.data_params
    if len(eval_data_list) == 0:
        for i in range(batch_size):
            random_data = get_random_data_all()
            eval_data = random_data.tail(70)
            eval_data_list.append(eval_data)

    for df in eval_data_list:
        df = keep_column_v2(df)
        x, y = get_xy_days_data_from_df(df, dp.feature_columns, dp.target_column, days)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list
