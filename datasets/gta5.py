import os, torch, numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage
from datasets.labelConverter import convertLabels
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from Extension.DAForm.countFreq import loadFrequenciesAndImages, normalizeFrequencies

class GTA5(Dataset):
    _label = 'labels'
    _images = 'images'
    
    def __init__(self, path:str, convertLabels:bool=True, transform=None, transformTarget=None, 
                    aug:BaseTransformation|list[BaseTransformation]=None, 
                    enableProbability:bool=False)-> None:
        """
        Loads the GTA5 dataset given the path to the directory containing the dataset.
        
        Args:
            path (str): path to the directory containing the dataset.
            convertLabels (bool, optional): whether to convert the labels or not. Defaults to True.
            transform (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
            transformTarget (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.
            aug (BaseTransformation|list[BaseTransformation], optional): augmentation to apply, if a list of augmentations is given
                they will all be applied. Defaults to None.
            enableProbability (bool, optional): if True, the images will be sampled based on the class frequencies. Defaults to False.
        """

        super(GTA5, self).__init__()
        self._convertLabels = convertLabels
        self._transform = transform
        self._transformTarget = transformTarget
        self._aug = aug
        self._enableProbability = enableProbability
        
        imagePath = os.path.join(path, GTA5._images)
        labelPath = os.path.join(path, GTA5._label)

        self._images = {i:[os.path.join(imagePath, image), os.path.join(labelPath, image)]
                        for i, image in enumerate(sorted(os.listdir(imagePath))) if image.endswith('.png')}
        
        if enableProbability is not None: 
            self._limit = enableProbability['limit'] if isinstance(enableProbability, dict) and 'limit' in enableProbability else 2500
            self._img_labels, self._freq = loadFrequenciesAndImages(self, num_classes=19, width=1280, height=720, useOurs=False,)
            self._freq = normalizeFrequencies(self._freq, T=enableProbability['T'] if isinstance(enableProbability, dict) and 'T' in enableProbability else 0.25)
            self.__initializeImagesCounter()
        
        
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
        if self._enableProbability:
            idx = self.__get_image_indexProb()
        
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
    
    
    def __get_image_indexProb(self, freq:list[float]=None)->int:
        """
        Returns the index of the image to be sampled based on the class frequencies
        using a multinomial distribution.
        
        Args:
            freq (list[float], optional): frequencies of the classes. If None, uses the class frequencies.
        
        Returns:
            idx (int): index of the image to be sampled.
        """
        if self.__counter >= self.__len__():
            self.__initializeImagesCounter()
        else:
            self.__counter += 1
        
        chosen = int(np.random.choice(a=sorted(self._img_labels[int(np.random.choice(list(self._img_labels.keys()), p=freq if freq else self._freq))])))
        
        while self.__counterDict[chosen] > self._limit:
            chosen = int(np.random.choice(a=sorted(self._img_labels[int(np.random.choice(list(self._img_labels.keys()), p=freq if freq else self._freq))])))
        self.__counterDict[chosen] += 1
        
        return chosen

    def __initializeImagesCounter(self)->None:
        """
        Initializes the counter for the images to be sampled based on the class frequencies
        using a dictionary where the key is the class label and the value is the number of times
        the image has been sampled.
        """
        self.__counter = 0
        self.__counterDict = {k:0 for k in self._img_labels.keys()}