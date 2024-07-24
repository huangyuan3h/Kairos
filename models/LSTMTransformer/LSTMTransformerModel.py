import torch

import torch.nn as nn


class LSTMAttentionTransformer(nn.Module):
    """结合 LSTM、Attention 和 Transformer 的时间序列预测模型."""

    def __init__(
        self,
        input_dim: int,  # 输入维度
        hidden_dim: int,  # LSTM 隐藏层维度
        num_layers: int,  # Transformer 层数
        num_heads: int,  # 注意力头数
        dropout_rate: float = 0.2,  # dropout 概率
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)  # LSTM 层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate)  # Attention 层

        # 使用 TransformerEncoderLayer 构建 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

        # Attention 机制
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch_size, seq_len, hidden_dim)

        # Transformer 编码
        transformer_out = self.transformer_encoder(attn_output.transpose(0, 1)).transpose(
            0, 1
        )  # (batch_size, seq_len, hidden_dim)

        # 使用最后一个时间步的输出
        out = self.fc(transformer_out[:, -1, :])  # (batch_size, 4)
        return out
