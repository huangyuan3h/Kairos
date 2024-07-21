import torch

import torch.nn as nn


class LSTMAttentionTransformer_v1_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(LSTMAttentionTransformer_v1_2, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                                    dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.lstm_linear = nn.Linear(hidden_dim, hidden_dim)
        self.transformer_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, 4)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # LSTM 编码
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)

        # Attention 机制
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.dropout(attn_output)

        # 线性变换和激活函数
        lstm_out = self.relu(self.lstm_linear(lstm_out[:, -1, :]))

        # Transformer 编码
        transformer_out = self.transformer_encoder(attn_output + lstm_out.unsqueeze(0))
        transformer_out = self.layer_norm(transformer_out)
        transformer_out = self.dropout(transformer_out)

        transformer_out = self.relu(self.transformer_linear(transformer_out[:, -1, :]))

        combined_output = torch.cat((lstm_out, transformer_out), dim=1)

        out = self.fc(combined_output)
        return out
