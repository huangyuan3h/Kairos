import torch
from torch.utils.data import DataLoader

from models.LSTMTransformer.LSTMTransformerModel import LSTMTransformerModel


def train_model(model: LSTMTransformerModel, dataloader: DataLoader, criterion, optimizer, num_epochs: int, save_path: str):
    model.train()
    for epoch in range(num_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    torch.save(model.state_dict(), save_path)