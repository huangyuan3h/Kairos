import torch
from torch.utils.data import DataLoader

def create_dataloader(dataset: torch.utils.data.IterableDataset, batch_size: int) -> DataLoader:
    """
    创建一个 DataLoader 用于加载股票数据。

    Args:
        dataset (torch.utils.data.IterableDataset): 数据集对象。
        batch_size (int): 批次大小。

    Returns:
        DataLoader: DataLoader 对象。
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,  # 丢弃最后一个不满批次大小的数据
    )
    return dataloader