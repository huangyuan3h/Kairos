import torch
import torch.nn as nn
from pytorch_forecasting.models import TemporalFusionTransformer


class TFTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.2):
        super(TFTModel, self).__init__()
        self.tft = TemporalFusionTransformer(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            dropout=dropout_rate,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            output_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tft(x)
