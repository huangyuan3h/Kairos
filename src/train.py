import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models import LSTMStockPredictor

# 超参数设置
seq_len = 20  # 序列长度
batch_size = 32
hidden_dim = 128
num_layers = 2
dropout = 0.3
learning_rate = 0.001
num_epochs = 100

# 数据加载
df = pd.read_csv("stock_data.csv")  # 请替换为您的数据文件路径
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
val_df = df[train_size:]
train_loader = get_dataloader(train_df, seq_len, batch_size)
val_loader = get_dataloader(val_df, seq_len, batch_size)

# 模型、优化器、损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMStockPredictor(input_dim=df.shape[1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 模型训练
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        confidence, expected_return_10, expected_return_5, max_drawdown, volatility = model(x)
        loss = criterion(confidence, y[:, 0]) + \
               criterion(expected_return_10, y[:, 1]) + \
               criterion(expected_return_5, y[:, 2]) + \
               criterion(max_drawdown, y[:, 3]) + \
               criterion(volatility, y[:, 4])
        loss.backward()
        optimizer.step()

    # 模型验证 (代码省略)

# 保存模型
torch.save(model.state_dict(), "trained_model.pth")