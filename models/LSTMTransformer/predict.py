import torch
import numpy as np


def predict(model, data, scaler):
    model.eval()
    with torch.no_grad():
        data_scaled = scaler.transform(data)
        x = torch.tensor(data_scaled[-60:], dtype=torch.float32).unsqueeze(0)
        predictions = model(x).numpy().flatten()
    return scaler.inverse_transform(np.column_stack([data[:, :-1], predictions.reshape(-1, 1)]))[:, -1]