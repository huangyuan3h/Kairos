import torch
from torch.utils.data import DataLoader

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 梯度裁剪
clip_value = 0.5 # 梯度裁剪值


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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 添加 ReduceLROnPlateau 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(num_epochs):
        for x, y in dataloader:
            x = x.float()
            y = y.float()
            x = x.to(device)  # 将输入数据移动到设备上
            y = y.to(device)  # 将目标数据移动到设备上

            optimizer.zero_grad()
            outputs = model(x)
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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, lr = {scheduler.get_last_lr()[0]}")

        # 在每个 epoch 结束后，根据 loss 更新学习率
        scheduler.step(loss)

        # 每 30 个 epoch 保存一次模型
        if (epoch + 1) % 30 == 0:
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, f'model_files/checkpoint_{epoch + 1}.pth')

    # 保存最终训练好的模型
    torch.save(model.state_dict(), save_path)
