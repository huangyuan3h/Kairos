import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # 将输入数据的维度从 (batch_size, seq_len, input_dim) 转换为 (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        # CNN 编码
        x = nn.functional.relu(self.conv1(x))
        # 将维度转换回 (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        # LSTM 编码
        lstm_out, _ = self.lstm(x)

        # 使用 LSTM 最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out
