
import torch

from sklearn.preprocessing import StandardScaler

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel


def predict(model: LSTMTransformerModel, data: torch.Tensor, scaler: StandardScaler, feature_columns: list) -> list:
    model.eval()
    with torch.no_grad():
        data_scaled = scaler.transform(data)
        x = torch.tensor(data_scaled[-60:, feature_columns], dtype=torch.float32).unsqueeze(0)
        predictions = model(x).numpy().flatten()
    return predictions.tolist()