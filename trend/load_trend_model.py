import torch
import os


def load_trend_model(model, path: str):
    """加载模型
    Args:
        model: 模型实例
        load_path: 模型文件路径
    Returns:
        模型实例
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"模型已加载：{path}")
    else:
        print(f"模型文件不存在，将创建新模型：{path}")
    return model
