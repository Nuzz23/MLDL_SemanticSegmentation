import torch
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation


class AdjustBrightness(BaseTransformation):
    """
    Class to perform brightness adjustment on images.
    """

    def __init__(self, p: float = 0.5, brightness_factor: float = 0.5) -> None:
        """
        Initializes the AdjustBrightness class.

        Args:
            p (float, optional): probability of applying the transformation. Defaults to 0.5.
            brightness_factor (float, optional): factor by which to adjust brightness. Defaults to 1.
                Note: brightness_factor must be between 0 and 1.
                A value of 0.5 will make the image half as bright, while a value of 1 will keep it unchanged.
                A value of 0 will make the image completely black.
        """
        if brightness_factor < 0 or brightness_factor > 1:
            raise ValueError("Brightness factor must be between 0 and 1")

        super(AdjustBrightness, self).__init__(p=p)
        self.__brightness_factor = brightness_factor
        
        
    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies brightness adjustment to the image if the probability is met.
        
        Args:
            image (torch.Tensor): image to be transformed.
            
        Returns:
            transformedImage (torch.Tensor): transformed image.
        """
        
        if torch.rand(1).item() < super().getProbability():
            return image * self.__brightness_factor 
            
        return image
    
    
    def __repr__(self) -> str:
        return f"AdjustBrightness(p={super().getProbability()}, brightness_factor={self.__brightness_factor})"
    
    def __str__(self) -> str:
        return self.__repr__()