import torch.nn as nn


class LSTMAttentionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(LSTMAttentionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                                    batch_first=True, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        # LSTM 编码
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, sequence_length, hidden_dim)

        # Attention 机制
        attn_output, _ = self.attention(lstm_out, lstm_out,
                                        lstm_out)  # attn_output: (batch_size, sequence_length, hidden_dim)

        # Transformer 编码
        transformer_out = self.transformer_encoder(attn_output)
        out = self.fc(transformer_out[:, -1, :])
        return out
