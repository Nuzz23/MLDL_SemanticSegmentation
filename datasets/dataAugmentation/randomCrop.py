import torch
import random
from torchvision.transforms import functional as F
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation

class RandomCrop(BaseTransformation):
    """
    Class to perform random crop on images and masks.
    """
    def __init__(self, p: float = 0.5, size=(1024, 512), padding=0, pad_if_needed=False):
        """
        Args:
            p (float): Probability of applying the transformation.
            size (tuple[int]): Desired output size (width, height).
            padding (int or sequence, optional): Optional padding on each border.
            pad_if_needed (bool): If True, pad image if smaller than desired size.
        """
        super().__init__(p=p)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random crop to image and mask.
        Args:
            image (torch.Tensor): Image to be cropped.
            mask (torch.Tensor): Mask to be cropped.
        Returns:
            Cropped image and mask.
        """
        if torch.rand(1).item() > self.getProbability():
            return image, mask

        # Optionally pad
        if self.padding > 0:
            image = F.pad(image, self.padding)
            mask = F.pad(mask, self.padding)

        _, h, w = image.shape
        th, tw = self.size[1], self.size[0]

        # Pad if needed
        if self.pad_if_needed and w < tw:
            pad = tw - w
            image = F.pad(image, (pad // 2, 0, pad - pad // 2, 0))
            mask = F.pad(mask, (pad // 2, 0, pad - pad // 2, 0))
        if self.pad_if_needed and h < th:
            pad = th - h
            image = F.pad(image, (0, pad // 2, 0, pad - pad // 2))
            mask = F.pad(mask, (0, pad // 2, 0, pad - pad // 2))

        _, h, w = image.shape
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        image = F.crop(image, i, j, th, tw)
        mask = F.crop(mask, i, j, th, tw)
        return image, mask

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(p={super().getProbability()}, size={self.size}, "
                f"padding={self.padding}, pad_if_needed={self.pad_if_needed})")

    def __str__(self):
        return self.__repr__()