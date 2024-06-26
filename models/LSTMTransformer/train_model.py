import torch
from torch.utils.data import DataLoader

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel

# 梯度裁剪
clip_value = 0.5  # 梯度裁剪值


def train_model(model: LSTMTransformerModel, dataloader: DataLoader, criterion, optimizer, num_epochs: int, save_path: str):
    model.train()
    for epoch in range(num_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)

            # 计算四个时间窗口的均值损失
            loss = criterion(outputs, y)

            if torch.isnan(loss):
                print("Loss is nan. Skipping this batch.")
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            loss.backward()
            optimizer.step()

            # 检查输出值
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Outputs contain NaN or Inf values. Skipping this batch.")
                continue

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), save_path)
