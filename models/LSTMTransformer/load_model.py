import torch
import os

def load_model(model, load_path):
    """加载模型
    Args:
        model: 模型实例
        load_path: 模型文件路径
    Returns:
        模型实例
    """
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f"模型已加载：{load_path}")
    else:
        print(f"模型文件不存在，将创建新模型：{load_path}")
    return model