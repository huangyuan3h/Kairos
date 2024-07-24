import pandas as pd
import numpy as np  # 导入 NumPy 库
from typing import Tuple
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

def filter_features_correlation(df: pd.DataFrame, threshold: float = 0.8) -> Tuple[pd.DataFrame, list, list]:
    """
    过滤掉相关性高于阈值的特征。

    Args:
        df (pd.DataFrame): 输入 DataFrame，包含所有特征。
        threshold (float, optional): 相关性阈值. Defaults to 0.8.

    Returns:
        Tuple[pd.DataFrame, list]: 过滤后的 DataFrame 和保留的特征名称列表。
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)  # 使用 np 调用 NumPy 函数
    )
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    filtered_df = df.drop(to_drop, axis=1)
    kept_features = filtered_df.columns.tolist()
    return filtered_df, kept_features, to_drop

def select_k_best_features(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, list]:
    """
    使用 SelectKBest 选择 K 个最佳特征。

    Args:
        X (pd.DataFrame): 特征 DataFrame。
        y (pd.Series): 目标变量 Series。
        k (int, optional): 选择的特征数量. Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, list]: 选择后的特征 DataFrame 和特征名称列表。
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_features), list(selected_features)