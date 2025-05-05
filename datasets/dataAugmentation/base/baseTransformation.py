from abc import ABC, abstractmethod
import torch

class BaseTransformation(ABC):
    def __init__(self, p: float = 0.5) -> None:
        """
        Base class for all transformations.
        
        Args:
            p (float, optional): Probability of applying the transformation. Defaults to 0.5.
        """
        self.__p = p

    def getProbability(self) -> float:
        """Returns the probability of applying the transformation.
        
        Returns:
            probability (float): Probability of applying the transformation.
        
        """
        return self.__p 

    @abstractmethod
    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to the image and mask.
        This method should be implemented by subclasses.
        
        Args:
            image (torch.Tensor): Image to be transformed.
            mask (torch.Tensor): Mask to be transformed.
            
        Returns:
            transformedImage (torch.Tensor): Transformed image.
            transformedMask (torch.Tensor): Transformed mask.
        """
        raise NotImplementedError("The transform method is not implemented for this class.")

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("The __repr__ method is not implemented for this class.")

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("The __str__ method is not implemented for this class.")
