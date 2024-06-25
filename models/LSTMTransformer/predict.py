import torch


def predict(model, data, scaler, feature_columns):
    model.eval()
    with torch.no_grad():
        data_scaled = scaler.transform(data)
        x = torch.tensor(data_scaled[-60:, feature_columns], dtype=torch.float32).unsqueeze(0)
        predictions = model(x).numpy().flatten()
    return predictions