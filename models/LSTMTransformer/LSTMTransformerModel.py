import torch.nn as nn


class LSTMTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, target_days):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.transformer = nn.Transformer(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, target_days)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out, lstm_out)
        out = self.fc(transformer_out[:, -1, :])
        return out