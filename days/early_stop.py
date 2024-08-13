import random

import torch

from data.data_merging.training_predict import get_random_v2_data_by_type
from days.days_parameter import get_days_config
from days.get_days_data import get_xy_days_data_from_df
from models.LSTMTransformer import LSTMAttentionTransformer
from models.standardize.FeatureStandardScaler import FeatureStandardScaler

SIZE_OF_VERIFY = 1000

validate_list = []


def evaluate_on_validation_set(model: LSTMAttentionTransformer, version: str, criterion, days):
    """
    使用验证集评估模型性能。

    Args:
        model
        version: 要评估的模型。
        criterion: 损失函数。
        days (int): 预测的天数。

    Returns:
        float: 验证集上的平均损失。
    """
    config = get_days_config(version)
    # 获取模型参数

    dp = config.data_params

    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    validation_loss = 0.0

    feature_scaler = FeatureStandardScaler(data_version="v2")
    feature_scaler.load_scaler()

    if len(validate_list) == 0:
        for _ in range(SIZE_OF_VERIFY):
            df = get_random_v2_data_by_type("verify")
            left_len = len(df) - 70
            for idx in range(3):
                idx = random.randint(0, left_len)
                target_df = df[idx:idx + 70]
                validate_list.append(target_df)

    with torch.no_grad():
        for idx in range(len(validate_list)):
            df = validate_list[idx]
            x, y_true = get_xy_days_data_from_df(df, dp.feature_columns, dp.target_column, days)
            y_true = torch.tensor(y_true).float().to(device)
            scaled_df = feature_scaler.transform(x)
            x = torch.tensor(scaled_df).float().to(device)
            x = x.unsqueeze(0)
            outputs = model(x)
            loss = criterion(outputs[0], y_true)
            validation_loss += loss.item()

    return validation_loss / len(validate_list)
