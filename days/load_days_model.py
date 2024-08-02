import torch
import os

def load_days_model(model, load_path,days =1):
    """加载模型
    Args:
        model: 模型实例
        load_path: 模型文件路径
    Returns:
        模型实例
    """
    final_path = load_path.format(days)
    if os.path.exists(final_path):
        model.load_state_dict(torch.load(final_path))
        print(f"模型已加载：{final_path}")
    else:
        print(f"模型文件不存在，将创建新模型：{final_path}")
    return model