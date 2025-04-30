import torch
from torchvision import transforms as T
from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
from datasets.gta5 import GTA5


# %% CityScapes
def transformation(width:int=1024, height:int=512):
  """
  Defines the transformation to be applied to the image

  Args:
    width (int): width of the image. Defaults to 1024.
    height (int): height of the image. Defaults to 512.

  Returns:
    transformations (tuple[T.Compose]): composed transformation.
      - transformations to the input tensor
      - transformations to the mask tensor
  """
  return T.Compose([
      T.Resize((height, width)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
      T.ToTensor(), # 2. Turn image values to between 0 & 1
      T.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                          std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
    ]),  T.Compose([
      T.Resize((height, width),interpolation=T.InterpolationMode.NEAREST), # 1. Reshape all images to 224x224 (though some models may require different sizes)
      T.ToTensor() , # 2. Turn image values to between 0 & 1
      T.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])


def loadData(batch_size: int, num_workers: int, pin_memory: bool, transform_train=None, transform_groundTruth=None):
    """
    Loads the CityScapes dataset given the path to the directory containing the dataset.

    Args:
        batch_size (int): batch size of the dataLoader.
        num_workers (int): number of workers of the dataLoader.
        pin_memory (bool): whether to pin memory of the dataLoader or not.
        transform (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
        targetTransfrom (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.

    Returns:
        train_loader (DataLoader): dataLoader for the training set.
        val_loader (DataLoader): dataLoader for the validation set.
    """
    return (DataLoader(CityScapes('data/Cityscapes/Cityspaces/images/train',
                                  transform=transform_train, targetTransfrom=transform_groundTruth),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
      DataLoader(CityScapes('data/Cityscapes/Cityspaces/images/val',
                            transform=transform_train, targetTransfrom=transform_groundTruth),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory))
    
    

# %% GTA5
def loadGTA5(batch_size:int, num_workers:int, pin_memory:bool, transform_train=None, transform_groundTruth=None)->DataLoader:
    """
    Loads the GTA5 dataset given the path to the directory containing the dataset.

    Args:
        batch_size (int): batch size of the dataLoader.
        num_workers (int): number of workers of the dataLoader.
        pin_memory (bool): whether to pin memory of the dataLoader or not.
        transform (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
        targetTransfrom (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.

    Returns:
        train_loader (DataLoader): dataLoader for the training set.
        val_loader (DataLoader): dataLoader for the validation set.
    """
    return DataLoader(GTA5('data/GTA5', transform=transform_train, targetTransfrom=transform_groundTruth),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)