import pandas as pd
import numpy as np  # 导入 NumPy 库
from typing import Tuple

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
