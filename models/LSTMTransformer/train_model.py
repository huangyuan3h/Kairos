import torch
from torch.utils.data import DataLoader

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel

# 梯度裁剪
clip_value = 0.5  # 梯度裁剪值


def train_model(model: LSTMTransformerModel, dataloader: DataLoader, criterion, optimizer, num_epochs: int,
                save_path: str):
    """
    训练模型。

    Args:
        model (LSTMTransformerModel): 要训练的模型。
        dataloader (DataLoader): 数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        num_epochs (int): 训练的 epoch 数。
        save_path (str): 保存训练好的模型路径。
    """
    model.train()
    for epoch in range(num_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x.float())
            y = y.float()
            # 计算损失
            loss = criterion(outputs, y)

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                print("Loss is nan. Skipping this batch.")
                continue

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            loss.backward()
            optimizer.step()

            # 检查输出值是否包含 NaN 或 Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Outputs contain NaN or Inf values. Skipping this batch.")
                continue

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # 每 10 个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, f'model_files/checkpoint_{epoch + 1}.pth')

    # 保存最终训练好的模型
    torch.save(model.state_dict(), save_path)
