from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from torchvision.transforms import ColorJitter as TorchColorJitter
import torch

class ColorJitter(BaseTransformation):
    """Applies color jitter to the image.
    Moreover, each transformation can be switched off by setting it to None or 0.
    - Brightness: it will adjust the brightness of the image, making it brighter or darker randomly, 
        the value tells only how drastic the change is, recommended between [0.5, 0.75].
    - Contrast: it will adjust the contrast of the image, making it more or less contrasted randomly,
        (ie colors will be more or less vivid), the value tells only how drastic the change is, recommended between [0.4, 0.85].
    - Saturation: it will adjust the saturation of the image, making it more or less saturated randomly,
        (ie colors will be more or less vivid), the value tells only how drastic the change is, recommended between [(1.5, 4), (2.5, 6)].
    - Hue: it will adjust the hue of the image, making it more or less RGB randomly,
        (ie colors will tend more to red, blue or green), the value tells only how drastic the change is, recommended between [0.1, 0.12].   
    
    
    **NOTE**: the following order is the order of transformation effects that are more realistic (ie more similar to the real world):
        1. brightness
        2. saturation
        3. contrast
        4. hue
    """
    def __init__(self, p: float = 0.5, brightness: float|None = 0.5, contrast: float|None = 0.6, saturation: float|tuple[float]|None = (2, 4.5), 
                    hue: float|None = 0.1) -> None:
        """ 
        Initializes the ColorJitter class.
        
        Args:
            p (float, optional): probability of applying the transformation. Defaults to 0.5.
            brightness (float|None, optional): brightness factor. Defaults to 0.5.
                Note: brightness factor must be between 0 and 1.
                A value of 0.5 will make the image half as bright, while a value of 1 will keep it unchanged.
                A value of 0 will make the image completely black.
                Note: can be None to disable brightness adjustment.
            contrast (float|None, optional): contrast factor. Defaults to 0.6.
                Note: can be None to disable contrast adjustment.
            saturation (float|tuple[float]|None, optional): saturation factor. Defaults to (2, 4.5).
                Note: can be None to disable saturation adjustment.
            hue (float|None, optional): hue factor. Defaults to 0.1.
                Note: can be None to disable hue adjustment.
        """
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