import torch


def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model