import torch

import torch.nn as nn


class LSTMAttentionTransformerClassify(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(LSTMAttentionTransformerClassify, self).__init__()

        # 1. LSTM 时序建模
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate)

        # 2. Attention 机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               dropout=dropout_rate,
                                               batch_first=True)

        # 3. Transformer 全局信息整合
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                               nhead=num_heads,
                                                               dropout=dropout_rate,
                                                               batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                         num_layers=num_layers)

        # 4. 输出层
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. LSTM 时序建模
        lstm_out, _ = self.lstm(x.float())  # (batch_size, seq_len, hidden_dim)

        # 2. Attention 机制
        attn_output, _ = self.attention(lstm_out, lstm_out,
                                        lstm_out)  # (batch_size, seq_len, hidden_dim)

        # 3. Transformer 全局信息整合
        transformer_out = self.transformer_encoder(
            attn_output + lstm_out)  # (batch_size, seq_len, hidden_dim)

        # 4. 使用最后一个时间步的输出进行分类
        out = self.fc(transformer_out[:, -1, :])  # (batch_size, 3)
        return out
