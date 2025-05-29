import os, torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage
from datasets.labelConverter import convertLabels
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
import pandas as pd
import json
import numpy as np
from Extension.DAForm.countFreq import loadFrequenciesAndImages

class GTA5Resampling(Dataset):
    _label = 'labels'
    _images = 'images'
    
    def __init__(self, path:str, convertLabels:bool=True, transform=None, transformTarget=None, 
                    aug:BaseTransformation|list[BaseTransformation]=None)-> None:
        """
        Loads the GTA5 dataset given the path to the directory containing the dataset.
        
        Args:
            path (str): path to the directory containing the dataset.
            convertLabels (bool, optional): whether to convert the labels or not. Defaults to True.
            transform (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
            transformTarget (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.
            aug (BaseTransformation|list[BaseTransformation], optional): augmentation to apply, if a list of augmentations is given
                they will all be applied. Defaults to None.
        """
        
        super(GTA5Resampling, self).__init__()
        self._convertLabels = convertLabels
        self._transform = transform
        self._transformTarget = transformTarget
        self._aug = aug
        
        self.img_labes, freq = loadFrequenciesAndImages()
        self.freq = torch.tensor(list(map(sorted(freq.items(), lambda x:x[0]))), dtype=torch.float32)
        self.img_labes2 = {k:list(v) for k, v in self.img_labes.items()}    
        
        imagePath = os.path.join(path, GTA5Resampling._images)
        labelPath = os.path.join(path, GTA5Resampling._label)
        
        
        
        
        self._images = {i:[os.path.join(imagePath, image), os.path.join(labelPath, image)]    
                        for i, image in enumerate(sorted(os.listdir(imagePath))) if image.endswith('.png')}
        

    def sample_images_with_class(self):
        """
        get the probability distribution of the classes and sample images based on the class frequencies.
    
        Args:"""
        sampled = False
        while not sampled: 
            c = np.random.choice(list(range(19)), p=self.freq.numpy())
            if len(self.img_labes[c]) > 0:
                img_list = np.random.choice(self.img_labes[c])
                #get a random image from Img_list and i remove it from the list
                img = img_list[np.random.randint(0, len(img_list))]
                # self._images.remove(img)
                return img
       
        

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
        idx = self.sample_images_with_class()
        
        image = decode_image(self._images[idx][0]).to(dtype=torch.uint8)
        mask =  decode_image(self._images[idx][1]).to(dtype=torch.uint8)

        if self._transform:
            image = self._transform(toPil(image))

        if self._transformTarget:
            mask = self._transformTarget(toPil(mask))
        
        if self._convertLabels:
            mask = convertLabels(mask[0], True).unsqueeze(0).to(dtype=torch.uint8)
            
        if self._aug:
            if isinstance(self._aug, list):
                for aug in self._aug:
                    image, mask = aug.transform(image, mask)
            else:
                image, mask = self._aug.transform(image, mask)

        return image, mask
        
        
    def __len__(self)->int:
        """
        Returns the number of images in the dataset.
        
        Returns:
            int: number of images in the dataset.
        """
        return len(self._images)
    
    def getImageLabels(self):
        """
        Returns the image labels.
        
        Returns:
            dict: dictionary containing the image labels.
        """
        return dict(self.img_labes.items())

    def startNewEpoch(self):
        """
        Sets the image labels.
        
        Args:
            img_labels (dict): dictionary containing the image labels.
        """
        self.img_labes = self.img_labes2.copy()
