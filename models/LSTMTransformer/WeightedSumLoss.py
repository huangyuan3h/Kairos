import numpy as np
import torch.nn as nn

predict_day = 10


class WeightedSumLoss(nn.Module):
    def __init__(self, weights: list):
        super(WeightedSumLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        losses = [nn.MSELoss()(outputs[:, i], targets[:, i]) for i in range(outputs.shape[1])]
        weighted_loss = sum([self.weights[i] * losses[i] for i in range(len(losses))])
        return weighted_loss


def get_weights():
    days = np.arange(predict_day)
    decay_rate = 0.9  # 控制衰减速度，可根据实验结果调整
    weights = decay_rate ** days
    weights = weights / weights.sum()
    return weights
