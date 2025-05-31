import os, torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage
import torchvision.transforms as T

class CityScapes(Dataset):
    _label = '_gtFine_labelTrainIds.png'
    _keyPathFinal = 'images'
    _valuePathFinal = 'gtFine'

    def __init__(self, path_dir: str, transform:T=None, targetTransform:T=None, deterministBehavior:bool=False) -> None:
        """
        Loads the CityScapes dataset given the path to the directory containing the dataset.

        Args:
            path_dir (str): path to the directory containing the dataset.
            transform (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
            targetTransform (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.
            deterministBehavior (bool, optional): if True, the index of the images will be returned so that the dataset can be used in a deterministic way. Defaults to False.
        """
        super(CityScapes, self).__init__()
        self._transform = transform
        self._targetTransform= targetTransform
        self._deterministBehavior = deterministBehavior

        prefix =  path_dir.split("/")[-1].strip()
        path_dir = '/'.join(path_dir.split("/")[:-2])

        prefixKey, prefixValue = os.path.join(path_dir, self._keyPathFinal, prefix), os.path.join(path_dir, self._valuePathFinal, prefix)

        # Note: now it creates a dictionary in the form index:[pathToSourceImage, pathToCorrespondingMask]
        self._images = {os.path.join(prefixKey, folder, photo):os.path.join(prefixValue, folder, '_'.join(photo.split("_")[:-1])) + CityScapes._label
                        for folder in os.listdir(prefixKey) for photo in os.listdir(os.path.join(prefixKey, folder)) if photo.endswith('.png')}

        self._images = {i:[key, self._images[key]] for i, key in enumerate(sorted(self._images.keys()))}


    def __getitem__(self, idx:int):
        """
        Loads the image and its corresponding label given the index.

        Args:
            idx (int): index of the image.

        Returns:
            images (list[torch.Tensor]): list containing the image and its corresponding label as torch tensor.
                - image (torch.Tensor): image as torch tensor.
                - mask label (torch.Tensor): mask label as torch tensor.
        """

        toPil = ToPILImage()

        image = decode_image(self._images[idx][0]).to(dtype=torch.uint8)
        mask =  decode_image(self._images[idx][1]).to(dtype=torch.uint8)

        if self._transform:
            image = self._transform(toPil(image))

        if self._targetTransform:
            mask = self._targetTransform(toPil(mask))

        return (image, mask) if not self._deterministBehavior else (image, mask, idx)


    def __len__(self)->int:
        """returns the number of images in the dataset.

        Returns:
            length (int) : number of images in the dataset.
        """
        return len(self._images)