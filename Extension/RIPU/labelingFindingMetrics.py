import torch
import numpy as np
from typing import Optional


def divideImageIntoSquares(image: torch.Tensor, k:int=32)->torch.Tensor:
    """
    Divides an image into k x k squares and returns a tensor with the indices of each square.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) where C is the number of channels,
                            H is the height and W is the width.
        k (int): Size of the square to divide the image into. Default is 32.
        
    Returns:
        mask (torch.Tensor): A tensor of shape (H//k, W//k) containing the indices of each square."""
    assert image.shape[-2] % k == 0 and image.shape[-1] % k == 0, "Image dimensions must be divisible by k"
    
    return torch.tensor(np.repeat(np.repeat(
        np.array(
            [[i*image.shape[-1]//k+j for j in range(image.shape[-1]//k)] for i in range(image.shape[-2]//k)],
            ), k, axis=0), k, axis=1))

def computeRegionBasedCriterionRA(image: torch.Tensor, mask: torch.Tensor, k:int=32, useLog2:bool=False)->torch.Tensor:
    """
    Computes the Region-based Criterion for Region Adversarial Learning (RIPU) on the given image and mask.
    The criterion is computed as the negative sum of the log probabilities of each pixel in the image,
    weighted by the number of pixels in each region defined by the mask.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) where C is the number of channels,
                            H is the height and W is the width.
        mask (torch.Tensor): Mask tensor of shape (H, W) where each pixel value represents a region.
        k (int): Size of the square to divide the image into. Default is 32.
        useLog2 (bool): If True, uses log base 2, otherwise uses natural log. Default is False.
        
    Returns:
        entropy (torch.Tensor): A tensor of shape (H, W) containing the computed entropy for each pixel in the image.
    """
    image = image.detach().clone().float()  # Assicurati che sia float
    if image.shape[0] == 1:
        image = image[0]
    assert image.dim() == 2, "Image must have 2 dimensions (h, w)"
    assert mask.dim() == 2, "Mask must have 2 dimensions (h, w)"
    assert image.shape == mask.shape, "Image and mask must have the same dimensions"


    log = np.log2 if useLog2 else np.log
    dim = float(mask[mask == 0].numel()) if k <= 0 else float(k**2)  # Evita int64

    for pixel in torch.unique(mask):
        locked = mask == pixel
        image[locked] = -sum((image[locked][mask[locked] == c].numel() / dim) * log((image[locked][mask[locked] == c].numel() / dim) + 1e-8) for c in torch.unique(image[locked]))

    return image


def computePixelEntropy(image: torch.Tensor, useLog2:bool=False)->torch.Tensor:
    """
    Computes the pixel-wise entropy of the image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) where C is the number of channels,
                            H is the height and W is the width.
        useLog2 (bool): If True, uses log base 2, otherwise uses natural log. Default is False.
        
    Returns:
        entropy (torch.Tensor): A tensor of shape (H, W) containing the computed entropy for each pixel in the image.
    """
    image = image.detach().clone().float()    
    assert image.dim() == 3, "Image must have 3 dimensions (C, H, W)"

    log = torch.log2 if useLog2 else torch.log
    return -(image * log(image + 1e-8)).sum(dim=0)


def computeU(pixel_entropy: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    """
    Computes the U criterion based on pixel entropy and a mask.
    The U criterion is computed as the mean pixel entropy for each unique pixel in the mask.
    
    Args:
        pixel_entropy (torch.Tensor): A tensor of shape (H, W) containing the pixel entropy values.
        mask (torch.Tensor): Mask tensor of shape (H, W) where each pixel value represents a region.
    
    Returns:
        pixel_entropy (torch.Tensor): A tensor of shape (H, W) containing the mean pixel entropy for each unique pixel in the mask."""
    for pixel in torch.unique(mask):
        locked = mask == pixel
        pixel_entropy[locked] = pixel_entropy[locked].mean()
    return pixel_entropy



def computeRAandPA(image, mask:Optional[torch.Tensor]=None, k:int=32, useLog2:bool=False, mu:float=-1, usePixelEntropy:bool=True):
    """
    Computes the Region-based Adversarial (RA) and Pixel Adversarial (PA) criteria for the given image.
    The RA criterion is computed as a weighted sum of the regional entropy and the pixel entropy,
    while the PA criterion is computed as the mean pixel entropy for each unique pixel in the mask.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) where C is the number of channels (19 classes),
                            H is the height and W is the width.
        mask (Optional[torch.Tensor]): Mask tensor of shape (H, W) where each pixel value represents a region.
                                    If None, the image will be divided into squares of size k x k.
        k (int): Size of the square to divide the image into. Default is 32.
        useLog2 (bool): If True, uses log base 2, otherwise uses natural log. Default is False.
        mu (float): Weighting factor for regional entropy and pixel entropy. Default is -1. if not in [0, 1], it will return the product of regional entropy and pixel entropy.
        usePixelEntropy (bool): If True, uses pixel entropy in the computation else compute U. Default is True.

    Returns:
        max (float): The maximum value of the computed RA and PA criteria across all pixels in the image.
    """
    mask = mask if mask is not None else divideImageIntoSquares(image, k=k)
    regional_entropy = computeRegionBasedCriterionRA(image.argmax(dim=0), mask, k=k, useLog2=useLog2)
    pixel_entropy = computePixelEntropy(image, useLog2=useLog2)
    U = pixel_entropy if usePixelEntropy else computeU(pixel_entropy, mask)

    return ((mu * regional_entropy + (1 - mu) * U) if 0<= mu <= 1 else regional_entropy*U).max()
