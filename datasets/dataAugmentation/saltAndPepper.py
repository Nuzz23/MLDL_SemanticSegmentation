import torch
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation

class SaltAndPepper(BaseTransformation):
    """
    Class to perform salt and pepper noise on images.
    """

    def __init__(self, p: float = 0.5, amount: float = 0.04) -> None:
        """
        Initializes the SaltAndPepper class.

        Args:
            p (float, optional): probability of applying the transformation. Defaults to 0.5.
            amount (float, optional): amount of salt and pepper noise to be added in percentage. Defaults to 0.04.
                Note: amount must be between 0 and 1. Half of the amount will be salt and half will be pepper.
        """
        if amount < 0 or amount > 1:
            raise ValueError("Amount must be between 0 and 1")
        
        super(SaltAndPepper, self).__init__(p=p)
        self.__amount = amount


    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies salt and pepper noise to the image if the probability is met.

        Args:
            image (torch.Tensor): image to be transformed.
            mask (torch.Tensor): mask to be transformed.

        Returns:
            transformedImage (torch.Tensor): transformed image.
            mask (torch.Tensor): transformed mask.
        """
        if torch.rand(1).item() < super().getProbability():
            return self.__salt_and_pepper(image), mask

        return image, mask
    
    
    def __salt_and_pepper(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies salt and pepper noise to the image.

        Args:
            image (torch.Tensor): image to be transformed.

        Returns:
            transformedImage (torch.Tensor): transformed image.
        """
        _, h, w = image.shape
        num = int(self.__amount * h * w * 0.5)
        
        # Add salt noise
        image[[torch.randint(0, i - 1, (num,)) for i in image.shape]] = 1

        # Add pepper noise
        image[[torch.randint(0, i - 1, (num,)) for i in image.shape]] = 0

        return image
    
    
    def __repr__(self) -> str:
        return f"SaltAndPepper(p={super().getProbability()}, amount={self.__amount})"
    
    def __str__(self) -> str:
        return self.__repr__()