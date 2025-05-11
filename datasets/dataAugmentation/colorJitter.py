from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from torchvision.transforms import ColorJitter as TorchColorJitter
import torch

class ColorJitter(BaseTransformation):
    def __init__(self, p: float = 0.5, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1) -> None:
        """
        Initializes the ColorJitter class.
        
        Args:
            p (float, optional): probability of applying the transformation. Defaults to 0.5.
            brightness (float, optional): brightness factor. Defaults to 0.2.
                Note: brightness factor must be between 0 and 1.
                A value of 0.5 will make the image half as bright, while a value of 1 will keep it unchanged.
                A value of 0 will make the image completely black.
            contrast (float, optional): contrast factor. Defaults to 0.2.
                Note: contrast factor must be between 0 and 1.
            saturation (float, optional): saturation factor. Defaults to 0.2.
                Note: saturation factor must be between 0 and 1.
            hue (float, optional): hue factor. Defaults to 0.1.
                Note: hue factor must be between -0.5 and 0.5.
        """
        
        if brightness < 0 or brightness > 1:
            raise ValueError("Brightness factor must be between 0 and 1")
        if contrast < 0 or contrast > 1:
            raise ValueError("Contrast factor must be between 0 and 1")
        if saturation < 0 or saturation > 1:
            raise ValueError("Saturation factor must be between 0 and 1")
        if hue < -0.5 or hue > 0.5:
            raise ValueError("Hue factor must be between -0.5 and 0.5")
        super().__init__(p)
        self.__brightness = brightness
        self.__contrast = contrast
        self.__saturation = saturation
        self.__hue = hue


    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies color jitter to the image if the probability is met.
        
        Args:
            image (torch.Tensor): image to be transformed.
            mask (torch.Tensor): mask to be transformed.
            
        Returns:
            transformedImage (torch.Tensor): transformed image.
            mask (torch.Tensor): transformed mask.
        """
        
        if torch.rand(1).item() < super().getProbability():
            return TorchColorJitter(brightness=self.__brightness, contrast=self.__contrast, saturation=self.__saturation, hue=self.__hue)(image), mask
        
        return image, mask
    
    def __repr__(self) -> str:
        return f"ColorJitter(p={super().getProbability()}, brightness={self.__brightness}, contrast={self.__contrast}, saturation={self.__saturation}, hue={self.__hue})"
    
    def __str__(self) -> str:
        return self.__repr__()
