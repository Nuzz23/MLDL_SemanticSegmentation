import os, torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage
from datasets.labelConverter import convertLabels


class GTA5(Dataset):
    _label = 'labels'
    _images = 'images'
    
    def __init__(self, path:str, convertLabels:bool=True, transform=None, transformTarget=None)-> None:
        """
        Loads the GTA5 dataset given the path to the directory containing the dataset.
        
        Args:
            path (str): path to the directory containing the dataset.
            convertLabels (bool, optional): whether to convert the labels or not. Defaults to True.
            transform (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
            transformTarget (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.
        """
        
        super(GTA5, self).__init__()
        self._convertLabels = convertLabels
        self._transform = transform
        self._transformTarget = transformTarget
        
        imagePath = os.path.join(path, GTA5._images)
        labelPath = os.path.join(path, GTA5._label)
        
        self._images = {i:[os.path.join(imagePath, image), os.path.join(labelPath, image)]    
                        for i, image in enumerate(sorted(os.listdir(imagePath))) if image.endswith('.png')}
        

    def __getitem__(self, idx:int)->torch.Tensor:
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

        if self._transformTarget:
            mask = self._transformTarget(toPil(mask))
        
        if self._convertLabels:
            mask = convertLabels(mask[0], True).unsqueeze(0).to(dtype=torch.uint8)

        return image, mask
        
        
    def __len__(self)->int:
        """
        Returns the number of images in the dataset.
        
        Returns:
            int: number of images in the dataset.
        """
        return len(self._images)
