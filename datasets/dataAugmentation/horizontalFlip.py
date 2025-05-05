import torch
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation

class HorizontalFlip(BaseTransformation):
    """
    Class to perform horizontal flip on images.
    """

    def __init__(self, p:float=0.5) -> None:
        """
        Initializes the HorizontalFlip class.

        Args:
            p (float, optional): probability of flipping the image. Defaults to 0.5.
        """
        super(HorizontalFlip, self).__init__(p=p)

    def transform(self, image:torch.Tensor) -> torch.Tensor:
        """
        Applies horizontal flip to the image if the probability is met.

        Args:
            image (torch.Tensor): image to be flipped.

        Returns:
            flippedImage (torch.Tensor): flipped image.
        """
        if torch.rand(1).item() < super().getProbability():
            return torch.flip(image, [2])
        
        return image
    
    
    def __repr__(self) -> str:
        return f"HorizontalFlip(p={super().getProbability()})"
    
    def __str__(self):
        return self.__repr__()