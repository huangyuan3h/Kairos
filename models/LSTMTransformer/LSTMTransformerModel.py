import torch

import torch.nn as nn


class LSTMAttentionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.0):
        super(LSTMAttentionTransformer, self).__init__()

        # CNN 特征提取
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # LSTM 时序建模
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate)

        # Attention 机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate)

        # Transformer 全局信息整合
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                               dropout=dropout_rate,  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN 特征提取
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, 128)

        # LSTM 时序建模
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

        # Attention 机制
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch_size, seq_len, hidden_dim)

        # Transformer 全局信息整合
        transformer_out = self.transformer_encoder(attn_output + lstm_out)  # (batch_size, seq_len, hidden_dim)

        # 使用最后一个时间步的输出
        out = self.fc(transformer_out[:, -1, :])  # (batch_size, 4)
        return out
