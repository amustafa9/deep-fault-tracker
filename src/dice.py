import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    This class implements the Dice Loss for 2D segmentation tasks 
    when the model outputs raw logits.

    Args:
        smooth: A small value to avoid division by zero. Defaults to 1e-6.
        p: The exponent for the denominator. Defaults to 2.
    """
    def __init__(self, smooth=1e-6, p=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, input, target):
        """
        Calculates the Dice Loss between the input logits and target.

        Args:
            input: The predicted logits (raw scores) from the model.
            target: The ground truth segmentation mask.

        Returns:
            The Dice Loss value.
        """
        assert input.size() == target.size(), "Input and target must have the same size"

        # Apply sigmoid activation to logits
        input = (torch.sigmoid(input) > 0.7).float()

        # Flatten the input and target tensors
        input = input.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Compute intersection and union
        intersection = (input * target).sum()
        union = input.pow(self.p).sum() + target.pow(self.p).sum() + self.smooth

        # Calculate Dice score
        score = 2. * intersection / union

        # Calculate Dice loss
        loss = 1 - score

        return loss