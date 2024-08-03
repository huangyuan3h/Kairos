import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedLSTMAttentionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.3):
        super(EnhancedLSTMAttentionTransformer, self).__init__()

        # 1. CNN for local feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2. Bi-LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate,
                            bidirectional=True)

        # 3. Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2,  # Bi-LSTM output dimension
                                               num_heads=num_heads,
                                               dropout=dropout_rate,
                                               batch_first=True)

        # 4. Transformer for global information integration
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2,
                                                               nhead=num_heads,
                                                               dropout=dropout_rate,
                                                               batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                         num_layers=num_layers)

        # 5. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim * 2, dropout=dropout_rate)

        # 6. Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

        # 7. Output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN
        x = F.relu(self.conv1(x.permute(0, 2, 1)))
        x = F.relu(self.conv2(x)).permute(0, 2, 1)

        # 2. Bi-LSTM
        lstm_out, _ = self.lstm(x.float())

        # 3. Attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 4. Transformer
        transformer_out = self.transformer_encoder(attn_output + lstm_out + self.pos_encoder(lstm_out))

        # 5. Batch Normalization
        transformer_out = self.bn(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 6. Output
        out = self.fc(transformer_out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding positional information to the inputs.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)