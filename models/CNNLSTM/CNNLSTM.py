import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        # CNN 特征提取
        x = x.permute(0, 2, 1)  # 将输入数据的维度从 (batch_size, seq_len, input_dim) 转换为 (batch_size, input_dim, seq_len)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # 将维度转换回 (batch_size, seq_len, input_dim)

        # LSTM 编码
        lstm_out, _ = self.lstm(x)

        # 使用 LSTM 最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out
