import torch
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
import torchvision.transforms as transforms

class RandomCrop(BaseTransformation):
    """
    Class to perform random crop on images and masks.
    """
    def __init__(self, p: float = 0.5, size:tuple[int]=(400, 800), pad_if_needed: bool=False):
        """
        Args:
            p (float): Probability of applying the transformation. Defaults to 0.5.
            size (tuple[int]): Desired output size (width, height). Defaults to (400, 800).
            pad_if_needed (bool): If True, pad image if smaller than desired size. Defaults to False.
        """
        super().__init__(p=p)
        self.__size = size
        self.__pad_if_needed = pad_if_needed

    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random crop to image and mask.
        Args:
            image (torch.Tensor): Image to be cropped.
            mask (torch.Tensor): Mask to be cropped.
            crop_size (tuple[int]): Desired output size (width, height). Defaults to (1024, 512).
        Returns:
            cropped_image (torch.Tensor): Cropped image.
            cropped_mask (torch.Tensor): Cropped mask.
        """

        if torch.rand(1).item() > self.getProbability():
            return image, mask
        
        resizeImage, resizeTarget = transforms.Resize(image.shape[-2:]), transforms.Resize(mask.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST)
        transform = transforms.RandomCrop(self.__size, pad_if_needed=self.__pad_if_needed, fill=255, padding_mode='constant')
        return resizeImage(transform(image)), resizeTarget(transform(mask))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(p={super().getProbability()}, size={self.__size}, "
                f"pad_if_needed={self.__pad_if_needed})")

    def __str__(self):
        return self.__repr__()