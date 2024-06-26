import torch.nn as nn
import torch


class LSTMTransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout_rate=0.2):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 4)  # 输出4个值，对应1天、3天、5天、10天的涨跌幅

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        transformer_out = self.transformer_encoder(lstm_out)
        out = self.fc(transformer_out[:, -1, :])
        return out