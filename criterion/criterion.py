import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


def get_criterion(config):
    if config.criterion.type == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss(
            reduction=config.criterion.reduction,
            beta=config.criterion.beta
        )
    elif config.criterion.type == 'RMSELoss':
        return RMSELoss(
            eps=config.criterion.eps,
            reduction=config.criterion.reduction
        )
    raise ValueError(f'Invalid criterion type: {config.criterion.ctype}')