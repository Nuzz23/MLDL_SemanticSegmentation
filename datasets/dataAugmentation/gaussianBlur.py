import torch 
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from torchvision.transforms import GaussianBlur as TorchGaussianBlur


class GaussianBlur(BaseTransformation):
    """
    Class to perform Gaussian blur on images.
    """

    def __init__(self, p: float = 0.5, kernel_size: int = 9, sigma:tuple[float] | float = 7) -> None:
        """
        Initializes the GaussianBlur class.

        Args:
            p (float, optional): probability of applying the transformation. Defaults to 0.5.
            kernel_size (int, optional): size of the Gaussian kernel. Defaults to 9.
                Note: kernel_size must be an odd number.
            sigma (tuple[float] | float, optional): standard deviation for Gaussian kernel. Defaults to 7.
        """
        if not kernel_size % 2:
            raise ValueError("Kernel size must be an odd number")
        
        super(GaussianBlur, self).__init__(p=p)
        self.__kernel_size = kernel_size
        self.__sigma = sigma
        
        
    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian blur to the image if the probability is met.
        
        Args:
            image (torch.Tensor): image to be transformed.
            
        Returns:
            transformedImage (torch.Tensor): transformed image.
        """ 
        
        if torch.rand(1).item() < super().getProbability():
            return TorchGaussianBlur(kernel_size=self.__kernel_size, sigma=self.__sigma)(image)
        
        return image
    
    def __repr__(self) -> str:
        return f"GaussianBlur(p={super().getProbability()}, kernel_size={self.__kernel_size}, sigma={self.__sigma})"
    
    def __str__(self) -> str:
        return self.__repr__()