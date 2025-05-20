import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth: A small constant added to the numerator and denominator to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted values (B, 1, H, W) after sigmoid
            targets: Ground truth labels (B, 1, H, W), binary (0 or 1)
        """
        # Apply sigmoid to input if it's logits
        #inputs = torch.sigmoid(inputs)
        
        # Flatten the inputs and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Compute the intersection and union
        intersection = torch.sum(inputs_flat * targets_flat)
        union = torch.sum(inputs_flat) + torch.sum(targets_flat)

        # Compute Dice score
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice score
        dice_loss = 1 - dice_score
        return dice_loss