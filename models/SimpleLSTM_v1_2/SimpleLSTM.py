import torch.nn as nn


class SimpleLSTM_v1_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(SimpleLSTM_v1_2, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        # LSTM 编码
        lstm_out, _ = self.lstm(x)

        out = self.fc(lstm_out[:, -1, :])
        return out
