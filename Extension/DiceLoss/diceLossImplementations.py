import torch

from Extension.DiceLoss.diceLoss import DiceLoss

def OnlyDiceLossBiSeNet(pred:torch.Tensor, truth:torch.Tensor, diceLoss:DiceLoss)->torch.Tensor:
    """ Computes the Dice loss for BiSeNet outputs.
    
    Args:
        pred (torch.Tensor): Predicted masks from BiSeNet, shape (B, C, H, W).
        truth (torch.Tensor): True masks, shape (B, H, W).
        diceLoss (DiceLoss): Instance of DiceLoss to compute the loss.
        
    Returns:
        diceLoss (torch.Tensor): Computed Dice loss.
    """
    return torch.mean(diceLoss(pred[i], truth, auxiliary=i!=0) for i in range(len(pred)))  if isinstance(pred, (tuple, list)) else diceLoss(pred, truth, auxiliary=False)


def DiceLossAndBiSeNetLoss(pred: torch.Tensor, truth: torch.Tensor, diceLoss: DiceLoss, biSeNetLoss_fn: torch.nn.Module, criterion: torch.nn.Module) -> torch.Tensor:
    """ Computes the combined loss of Dice loss and BiSeNet loss.
    
    Args:
        pred (torch.Tensor): Predicted masks from BiSeNet, shape (B, C, H, W).
        truth (torch.Tensor): True masks, shape (B, H, W).
        diceLoss (DiceLoss): Instance of DiceLoss to compute the Dice loss.
        biSeNetLoss_fn (torch.nn.Module): Instance of BiSeNet loss to compute the BiSeNet loss.
        criterion (torch.nn.Module): Instance of the criterion to compute the final loss.
        
    Returns:
        combined_loss (torch.Tensor): Combined loss of Dice and BiSeNet losses.
    """
    return OnlyDiceLossBiSeNet(pred, truth, diceLoss) + biSeNetLoss_fn(pred, truth, criterion)