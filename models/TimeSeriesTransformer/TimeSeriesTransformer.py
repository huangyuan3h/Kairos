import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """时间序列 Transformer 模型 (类似 TimeFM 结构)."""

    def __init__(
            self,
            input_dim: int,  # 输入维度
            hidden_dim: int,  # 隐藏层维度
            num_layers: int,  # Transformer 层数
            num_heads: int,  # 注意力头数
            dropout_rate: float = 0.2,  # dropout 概率
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)  # 输入映射层

        # 使用 TransformerEncoderLayer 构建 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Transformer 编码
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)  # (batch_size, seq_len, hidden_dim)

        # 使用最后一个时间步的输出
        out = self.fc(x[:, -1, :])  # (batch_size, 4)
        return out
