import torch
from torch.utils.data import DataLoader

from days.StockDatasetDays import steps_per_epoch
from models.LSTMTransformer.LSTMTransformerModel import LSTMAttentionTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from operation.operation_early_stop import evaluate_operation_on_validation_set
from operation.operation_parameter import get_operation_config

# 梯度裁剪
clip_value = 0.5

patience = 5

# 连续没有改善的epoch数
epochs_without_improvement_threshold = 2


def train_operation_model(model: LSTMAttentionTransformer, version: str, dataloader: DataLoader, criterion, optimizer,
                          days=1):
    """
    训练模型。

    Args:
        model (LSTMTransformerModel): 要训练的模型。
        version (str): model version
        dataloader (DataLoader): 数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        days: training day
    """
    config = get_operation_config(version)
    tp = config.training_params
    model.train()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 添加 ReduceLROnPlateau 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(tp.num_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in dataloader:
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
            epoch_loss = epoch_loss + loss.item()

            # 检查输出值是否包含 NaN 或 Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Outputs contain NaN or Inf values. Skipping this batch.")
                continue
        epoch_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1}/{tp.num_epochs}, Loss: {epoch_loss}, lr = {scheduler.get_last_lr()[0]}")

        # 在每个 epoch 结束后评估验证集
        validation_loss = evaluate_operation_on_validation_set(model, version, criterion, days)
        print(f"Epoch {epoch + 1}/{tp.num_epochs}, Validation Loss: {validation_loss}")

        final_path = tp.model_save_path.format(days)
        # Early stopping
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), final_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= epochs_without_improvement_threshold:
                print(f"No improvement in {epochs_without_improvement_threshold} epochs. Reloading best model.")
                # 加载最好的模型
                model.load_state_dict(torch.load(final_path))
                # 重置 epochs_without_improvement
                epochs_without_improvement = 0
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                return

        # 更新学习率
        scheduler.step(validation_loss)
