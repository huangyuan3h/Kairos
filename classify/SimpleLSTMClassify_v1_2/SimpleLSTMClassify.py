import torch
import torch.nn as nn

class SimpleLSTMClassify_v1_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.0):
        super(SimpleLSTMClassify_v1_2, self).__init__()

        # LSTM 时序建模
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 3)  # 修改为3分类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM 时序建模
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

        # 使用最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])  # (batch_size, 3)
        return out