
import torch

from sklearn.preprocessing import StandardScaler

from models.LSTMTransformer.LSTMTransformerModel import LSTMAttentionTransformer


def predict(model: LSTMAttentionTransformer, data: torch.Tensor, scaler: StandardScaler, feature_columns: list) -> list:
    model.eval()
    with torch.no_grad():
        data_scaled = scaler.transform(data)
        x = torch.tensor(data_scaled[-60:, feature_columns], dtype=torch.float32).unsqueeze(0)
        predictions = model(x).numpy().flatten()
    return predictions.tolist()