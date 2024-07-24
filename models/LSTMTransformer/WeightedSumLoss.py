import torch.nn as nn


class WeightedSumLoss(nn.Module):
    def __init__(self, weights: list):
        super(WeightedSumLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        losses = [nn.MSELoss()(outputs[:, i], targets[:, i]) for i in range(outputs.shape[1])]
        weighted_loss = sum([self.weights[i] * losses[i] for i in range(len(losses))])
        return weighted_loss