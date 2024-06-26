import torch.nn as nn
import torch


class LSTMTransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, target_days: int):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, target_days)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # 使用TransformerEncoder而非nn.Transformer，避免手动进行key, query, value的线性变换
        transformer_out = self.transformer_encoder(lstm_out)
        out = self.fc(transformer_out[:, -1, :])
        return out