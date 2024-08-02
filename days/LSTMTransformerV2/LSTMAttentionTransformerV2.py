import torch
import torch.nn as nn


class LSTMAttentionTransformerV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(LSTMAttentionTransformerV2, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=3,
                            batch_first=True,
                            dropout=dropout_rate)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               dropout=dropout_rate,
                                               batch_first=True)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                               nhead=num_heads,
                                                               dropout=dropout_rate,
                                                               batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                         num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x.float())
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)

        transformer_out = self.transformer_encoder(attn_output + lstm_out)
        transformer_out = self.layer_norm(transformer_out)

        out = self.fc(transformer_out[:, -1, :])
        return out
