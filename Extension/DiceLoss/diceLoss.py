import torch
from torch.nn import functional as F


class DiceLoss(torch.nn.Module):
    """    Dice Loss for semantic segmentation tasks.
        This loss is particularly useful for imbalanced datasets, as it focuses on the overlap between the predicted and true masks.

        Args:
            numClasses (int, optional): Number of classes in the segmentation task. Defaults to 19.
            smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-6.
            ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to 255.
            auxiliaryWeights (float, optional): Weight for auxiliary losses. Defaults to 0.4.
        """

    def __init__(self, numClasses:int=19, smooth:float=1e-6, ignore_index:int=255, auxiliaryWeights:float=0.4, *args, **kwargs)->None:
        """ Initializes the DiceLoss module.
        This loss is particularly useful for imbalanced datasets, as it focuses on the overlap between the predicted and true masks.

        Args:
            numClasses (int, optional): Number of classes in the segmentation task. Defaults to 19.
            smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-6.
            ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to 255.
            auxiliaryWeights (float, optional): Weight for auxiliary losses. Defaults to 0.4.
        """
        super().__init__(*args, **kwargs)
        self.__numClasses = numClasses
        self.__smooth = smooth
        self.__ignore_index = ignore_index
        self.__auxiliaryWeights = auxiliaryWeights
    
    
    def forward(self, pred:torch.Tensor, truth:torch.Tensor, auxiliary:bool=False)->torch.Tensor:
        """ Computes the Dice loss between predicted and true masks.
        Args:
            pred (torch.Tensor): Predicted masks, shape (B, C, H, W) or (B, H, W) for semantic segmentation tasks.
            truth (torch.Tensor): True masks, shape (B, C, H, W) or (B, H, W) for semantic segmentation tasks.
        Returns:
            dice_loss (torch.Tensor): Computed Dice loss."""
        pred, truth, weight = pred.clone(), truth.clone(), self.__auxiliaryWeights if auxiliary else 1.0
        pred, truth  = pred.argmax(dim=1) if pred.dim() == 4 else pred, truth.squeeze(1) if truth.dim() == 4 else truth
        pred[pred == self.__ignore_index], truth[truth==self.__ignore_index] = self.__numClasses, self.__numClasses
        
        pred = F.one_hot(pred, num_classes=self.__numClasses+1).permute(0, 3, 1, 2).float()[:, :-1]  
        truth = F.one_hot(truth, num_classes=self.__numClasses+1).permute(0, 3, 1, 2).float()[:, :-1]  
        
        intersection = (pred * truth).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + truth.sum(dim=(2, 3)) - intersection

        return (1 - ((2 * intersection + self.__smooth) / (union + self.__smooth)).mean())* weight
    
    def __repr__(self)->str:
        """ Returns a string representation of the DiceLoss module."""
        return f"{self.__class__.__name__}(numClasses={self.__numClasses}, smooth={self.__smooth}, ignore_index={self.__ignore_index}, auxiliaryWeights={self.__auxiliaryWeights})"
    
    def __str__(self):
        return self.__repr__()