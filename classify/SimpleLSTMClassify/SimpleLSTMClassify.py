import torch
import torch.nn as nn


class SimpleLSTMClassify(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.1):
        super(SimpleLSTMClassify, self).__init__()

        # LSTM 时序建模
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM 时序建模
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        # 使用最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])  # (batch_size, 3)
        return out
