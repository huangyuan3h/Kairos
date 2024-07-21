import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.5):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 4)  # 双向 LSTM 输出是 hidden_dim 的两倍
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将输入数据的维度从 (batch_size, seq_len, input_dim) 转换为 (batch_size, input_dim, seq_len)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # 将维度转换回 (batch_size, seq_len, input_dim)

        # LSTM 编码
        lstm_out, _ = self.lstm(x)

        # 使用 LSTM 最后一个时间步的输出
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out
