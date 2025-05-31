import torch
import torch.nn as nn
import torch.nn.functional as F



class LocalConsistentLoss(nn.Module):
    """
    Local Consistent Loss for semantic segmentation tasks.
    This loss function computes the local consistency of predictions
    by comparing each pixel with its 3x3 neighborhood.
    It is designed to encourage the model to produce consistent predictions
    across neighboring pixels, which is particularly useful in semantic segmentation tasks."""
    def __init__(self):
        super(LocalConsistentLoss, self).__init__()

    def forward(self, x):
        """
        Computes local consistent loss using 3x3 neighborhoods efficiently.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size,
                            C is the number of channels, H is the height, and W is the width.
        Returns:
            loss (torch.Tensor): The computed local consistent loss.
        """
        N, C, H, W = x.shape
        return torch.abs(x - F.unfold(x, kernel_size=3, padding=1).view(N, C, 9, H, W).mean(dim=2)).mean()

class NegativeLearningLoss(nn.Module):
    """
    Negative Learning Loss for semantic segmentation tasks.
    This loss function penalizes predictions with confidence below a certain threshold.
    It is designed to encourage the model to focus on confident predictions,
    while discouraging low-confidence predictions.
    """
    def __init__(self, threshold=0.05):
        """
        Initializes the Negative Learning Loss with a specified threshold.
        The threshold determines the confidence level below which predictions are penalized.

        Args:
            threshold (float, optional): The confidence threshold for penalizing predictions. Defaults to 0.05.
        """
        super(NegativeLearningLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        """
        Computes the Negative Learning Loss.
        Penalizes predictions with confidence below `threshold`.
        
        Args:
            predict (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size,
                                    C is the number of classes, H is the height, and W is the width.
        """
        mask = (predict < self.threshold)          
        return torch.sum(mask * -torch.log(torch.clamp(1 - predict, min=1e-8))) / (torch.sum(mask) + 1e-8) 
