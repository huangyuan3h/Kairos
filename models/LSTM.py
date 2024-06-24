import torch.nn as nn
import torch


class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_confidence = nn.Linear(hidden_dim, 1)
        self.fc_return = nn.Linear(hidden_dim, 1)
        self.fc_drawdown = nn.Linear(hidden_dim, 1)
        self.fc_volatility = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        confidence = torch.sigmoid(self.fc_confidence(out[:, -1, :]))
        expected_return_10 = self.fc_return(out[:, -1, :])
        expected_return_5 = self.fc_return(out[:, -1, :])
        max_drawdown = self.fc_drawdown(out[:, -1, :])
        volatility = torch.exp(self.fc_volatility(out[:, -1, :]))
        return confidence, expected_return_10, expected_return_5, max_drawdown, volatility