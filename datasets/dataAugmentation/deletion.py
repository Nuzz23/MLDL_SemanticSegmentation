import torch
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from torchvision.transforms import RandomErasing as TorchRandomErasing


class RandomErasing(BaseTransformation):
    """
    Class to perform random erasing on images.
    """

    def __init__(self, p: float = 0.5, scale: tuple[float] = (0.02, 0.33), ratio: tuple[float] = (0.3, 3.3),
                 deleteInMask:bool=False) -> None:
        """
        Initializes the RandomErasing class.
        
        Args:
            p (float, optional): probability of applying the transformation. Defaults to 0.5.
            scale (tuple[float], optional): scale of the erased area. Defaults to (0.02, 0.33).
                Note: scale must be between 0 and 1.
            ratio (tuple[float], optional): aspect ratio of the erased area. Defaults to (0.3, 3.3).
            deleteInMask (bool, optional): if True, the erased area in the mask will be set to 255. Defaults to True.
        """
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale must be between 0 and 1")
        
        super(RandomErasing, self).__init__(p=p)
        self.__scale = scale
        self.__ratio = ratio
        self.__deleteInMask = deleteInMask
        
        
    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies random erasing to the image if the probability is met.
        
        Args:
            image (torch.Tensor): image to be transformed.
            mask (torch.Tensor): mask to be transformed.

        Returns:
            transformedImage (torch.Tensor): transformed image.
            mask (torch.Tensor): transformed mask.
        """
        if torch.rand(1).item() < super().getProbability():
            if self.__deleteInMask:
                eraser = TorchRandomErasing(scale=self.__scale, ratio=self.__ratio, p=1, inplace=False)
                params = eraser.get_params(image, scale=self.__scale, ratio=self.__ratio, value=0)
                x, y, h, w, v = params
                erased_image = image.clone()
                erased_image[..., y:y + h, x:x + w] = v
                erased_mask = mask.clone()
                erased_mask[..., y:y + h, x:x + w] = 255
                return erased_image, erased_mask
            else:
                return TorchRandomErasing(scale=self.__scale, ratio=self.__ratio, p=1)(image), mask

        return image, mask
    
    
    def __repr__(self) -> str:
        return f"RandomErasing(p={super().getProbability()}, scale={self.__scale}, ratio={self.__ratio}, deleteInMask={self.__deleteInMask})"
    
    def __str__(self) -> str:
        return self.__repr__()