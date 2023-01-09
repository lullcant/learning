import torch
import numpy as np
from torch import Tensor
def dice_coeff(pred,target,epsilon=1e-6,reduce_batch_first: bool = False):
    sum_dim = (-1,-2) if pred.dim() ==2 or not reduce_batch_first else (-1,-2,-3)
    intersection = (2* pred*target).sum(dim=sum_dim)
    union = pred.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    return (intersection+epsilon/union+epsilon).mean()##防除法无意义。

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon=1e-6)

def dice_loss(pred,target,muticlass=False):
    if muticlass:
        return 1 - multiclass_dice_coeff(pred,target,reduce_batch_first=True,epsilon=1e-6)
    else:
        return 1 - dice_coeff(pred,target,epsilon=1e-6,reduce_batch_first=True)