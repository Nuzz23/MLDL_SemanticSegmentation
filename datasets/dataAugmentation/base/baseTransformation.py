from abc import ABC, abstractmethod
import torch

class BaseTransformation(ABC):
    def __init__(self, p: float = 0.0) -> None:
        """
        Base class for all transformations.
        
        Args:
            p (float, optional): Probability of applying the transformation. Defaults to 0.0.
        """
        self.__p = p

    def getProbability(self) -> float:
        """Returns the probability of applying the transformation.
        
        Returns:
            probability (float): Probability of applying the transformation.
        
        """
        return self.__p  # Attributo privato con accesso in sola lettura

    @abstractmethod
    def transform(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The transform method is not implemented for this class.")

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("The __repr__ method is not implemented for this class.")

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("The __str__ method is not implemented for this class.")
