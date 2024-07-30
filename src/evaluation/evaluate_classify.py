from classify.get_classify_data import get_classify_xy_data_from_df
from classify.predict import ModelPredictorClassify
from data.data_merging.merge_data_v2 import get_random_data_all, keep_column_v2
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from src.training.parameter import get_config


def evaluate_classify_model(model_name: str, get_data_func) -> dict:
    """
    使用提供的数据获取函数评估分类模型的准确度和其他指标。

    Args:
        model_name (str): 模型名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        dict: 包含评估指标的字典。
    """
    predictor = ModelPredictorClassify(model_name)
    X, y_true = get_data_func(model_name)

    # 使用模型进行预测
    predictions = []
    for x in X:
        prediction = predictor.predict(x)
        predictions.append(prediction['predict_class'].values[0])  # 获取预测值

    y_true = [yt[0] for yt in y_true]  # 转换为一维列表
    y_true = torch.tensor(y_true)
    predictions = torch.tensor(predictions)

    # 计算指标
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='weighted')
    recall = recall_score(y_true, predictions, average='weighted')
    f1 = f1_score(y_true, predictions, average='weighted')
    cm = confusion_matrix(y_true, predictions)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
    }


def compare_classify_models(model_1_name: str, model_2_name: str, get_data_func) -> pd.DataFrame:
    """
    使用提供的数据获取函数比较两个分类模型的性能。

    Args:
        model_1_name (str): 第一个模型的名称。
        model_2_name (str): 第二个模型的名称。
        get_data_func (function): 用于获取数据的函数，返回两个列表：X 和 y。

    Returns:
        pd.DataFrame: 包含两个模型评估指标的 DataFrame。
    """
    results = {}
    for model_name in [model_1_name, model_2_name]:
        results[model_name] = evaluate_classify_model(model_name, get_data_func)
    return pd.DataFrame.from_dict(results, orient='index')


eval_data_list = []

batch_size = 1000

def get_my_data_classify(model_name="v1"):
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
        x, y = get_classify_xy_data_from_df(df, dp.feature_columns, dp.target_column)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list
