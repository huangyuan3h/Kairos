import torch
from torch.utils.data import DataLoader

from days.days_parameter import get_days_config
from days.early_stop import evaluate_on_validation_set
from models.LSTMTransformer.LSTMTransformerModel import LSTMAttentionTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 梯度裁剪
clip_value = 0.5

patience = 60


def train_days_model(version: str, dataloader: DataLoader, criterion, optimizer, days=1):
    """
    训练模型。

    Args:
        version (str): model version
        dataloader (DataLoader): 数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        days: training day
    """
    config = get_days_config(version)
    # 获取模型参数
    tp = config.training_params
    model = config.Model

    model.train()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 添加 ReduceLROnPlateau 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    best_loss = float('inf')
    epochs_without_improvement = 0
    epoch_loss = 100000  # very large number

    for epoch in range(tp.num_epochs):

        for step, (x, y) in enumerate(dataloader):
            x = x.float().to(device)
            y = y.float().to(device)

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
            epoch_loss = loss.item()

            # 检查输出值是否包含 NaN 或 Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Outputs contain NaN or Inf values. Skipping this batch.")
                continue

        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}/{tp.num_epochs}, Loss: {epoch_loss}, lr = {scheduler.get_last_lr()[0]}")

        # 在每个 epoch 结束后评估验证集
        validation_loss = evaluate_on_validation_set(version, criterion, days)
        print(f"Epoch {epoch + 1}/{tp.num_epochs}, Validation Loss: {validation_loss}")

        # Early stopping
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            final_path = tp.model_save_path.format(days)
            torch.save(model.state_dict(), final_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                return

        # 更新学习率
        scheduler.step(validation_loss)
