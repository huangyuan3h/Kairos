

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.data_merging.merge_data_v2 import get_random_data_all, keep_column_v2
from models.LSTMTransformer.WeightedSumLoss import get_weights
from models.LSTMTransformer.get_data import get_xy_data_from_df
from models.LSTMTransformer.predict import ModelPredictor
from src.training.parameter import get_config


def evaluate_model(model_name: str, get_data_func) -> dict:
    """
    使用提供的数据获取函数评估模型的准确度，并考虑不同时间步的权重。

    Args:
        model_name (str): 模型名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        dict: 包含评估指标的字典。
    """
    predictor = ModelPredictor(model_name)
    X, y_true = get_data_func(model_name)
    weights = get_weights()
    weights = torch.tensor(weights)

    # 使用模型进行预测
    predictions = []
    for x in X:
        prediction = predictor.predict(x)
        predictions.append(prediction[0])  # 获取预测值
    predictions = torch.stack([torch.tensor(p) for p in predictions])
    y_true = torch.tensor(y_true)


    # 计算加权指标
    weighted_mae = (torch.abs(predictions - y_true) * weights).mean()
    weighted_mse = ((predictions - y_true) ** 2 * weights).mean()
    weighted_rmse = weighted_mse ** 0.5
    weighted_r2 = r2_score(y_true.view(-1), predictions.view(-1), sample_weight=weights.repeat(len(y_true)))

    # 计算每个时间步的指标
    mae_per_step = [mean_absolute_error(y_true[:, i], predictions[:, i]) for i in range(predictions.shape[1])]
    mse_per_step = [mean_squared_error(y_true[:, i], predictions[:, i]) for i in range(predictions.shape[1])]
    rmse_per_step = [mse ** 0.5 for mse in mse_per_step]

    return {
        "Weighted MAE": weighted_mae.item(),
        "Weighted MSE": weighted_mse.item(),
        "Weighted RMSE": weighted_rmse.item(),
        "Weighted R2": weighted_r2,
        "MAE per Step": mae_per_step,
        "MSE per Step": mse_per_step,
        "RMSE per Step": rmse_per_step,
    }


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


eval_data_list = []

batch_size = 500


def get_my_data(model_name="v1"):
    x_list = []
    y_list = []

    config = get_config(model_name)
    # 获取模型参数
    dp = config.data_params
    data_version = config.data
    if len(eval_data_list) == 0:
        for i in range(batch_size):
            random_data = get_random_data_all()
            eval_data = random_data.tail(70)
            eval_data_list.append(eval_data)

    for df in eval_data_list:
        df = keep_column_v2(df)
        x, y = get_xy_data_from_df(df, dp.feature_columns, dp.target_column)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list
