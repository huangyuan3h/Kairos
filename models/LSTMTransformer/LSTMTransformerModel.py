import torch.nn as nn
import torch


class LSTMTransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, target_days: int):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.transformer = nn.Transformer(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, target_days)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out, lstm_out)
        out = self.fc(transformer_out[:, -1, :])
        return out