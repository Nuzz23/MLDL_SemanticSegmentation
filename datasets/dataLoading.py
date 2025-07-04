import torch
from torchvision import transforms as T
from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
from datasets.gta5 import GTA5
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation


# %% CityScapes
def transformationCityScapes(width:int=1024, height:int=512)->tuple[T.Compose]:
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
      T.Resize((height, width)), 
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]),  T.Compose([
      T.Resize((height, width),interpolation=T.InterpolationMode.NEAREST),
      T.ToTensor(), 
      T.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

def loadData(batch_size: int, num_workers: int, pin_memory: bool, transform_train: T=None, transform_groundTruth: T=None)->tuple[DataLoader, DataLoader]:
    """
    Loads the CityScapes dataset given the path to the directory containing the dataset.

    Args:
        batch_size (int): batch size of the dataLoader.
        num_workers (int): number of workers of the dataLoader.
        pin_memory (bool): whether to pin memory of the dataLoader or not.
        transform_train (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
        transform_groundTruth (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.

    Returns:
        train_loader (DataLoader): dataLoader for the training set.
        val_loader (DataLoader): dataLoader for the validation set.
    """
    return (DataLoader(CityScapes('data/Cityscapes/Cityspaces/images/train',
                                  transform=transform_train, targetTransform=transform_groundTruth),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
      DataLoader(CityScapes('data/Cityscapes/Cityspaces/images/val',
                            transform=transform_train, targetTransform=transform_groundTruth),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory))


# %% GTA5
def transformationGTA5(width:int=1280, height:int=720)->tuple[T.Compose]:
  """
  Defines the transformation to be applied to the image

  Args:
    width (int): width of the image. Defaults to 1280.
    height (int): height of the image. Defaults to 720.

  Returns:
    transformations (tuple[T.Compose]): composed transformation.
      - transformations to the input tensor
      - transformations to the mask tensor
  """
  return T.Compose([
      T.Resize((height, width)), 
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]),  T.Compose([
      T.Resize((height, width),interpolation=T.InterpolationMode.NEAREST),
      T.ToTensor(), 
      T.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])


def loadGTA5(batch_size:int, num_workers:int, pin_memory:bool, transform_train=None, transform_groundTruth=None,
              augmentation:BaseTransformation|None=None, convertLabels:bool=True, enableProbability:bool|dict[str, float]=False)->DataLoader:
    """
    Loads the GTA5 dataset given the path to the directory containing the dataset.

    Args:
        batch_size (int): batch size of the dataLoader.
        num_workers (int): number of workers of the dataLoader.
        pin_memory (bool): whether to pin memory of the dataLoader or not.
        transform_train (torchvision.transforms, optional): transformations to apply to the images. Defaults to None.
        transform_groundTruth (torchvision.transforms, optional): transformations to apply to the masks. Defaults to None.
        augmentation (BaseTransformation|None, optional): augmentation to apply, if a list of augmentations is given
            they will all be applied. Defaults to None.
        convertLabels (bool, optional): whether to convert the labels or not. Defaults to True.
        enableProbability (bool|dict[str, float], optional): if a dict is given one of the following keys should be given:
            - 'T': normalization temperature for the sampling probabilities. If not given, defaults to 0.25
            - 'limit': maximum number of times a figure is allowed to be sampled per epoch. If not given, no limit is applied.
        if True, we will use the default hyperparameters. Defaults to False.

    Returns:
        train_loader (DataLoader): dataLoader for the training set.
        val_loader (DataLoader): dataLoader for the validation set.
    """
    return DataLoader(GTA5('data/GTA5', transform=transform_train, transformTarget=transform_groundTruth, aug=augmentation, convertLabels=convertLabels, enableProbability=enableProbability),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
    
# %% Basic Transformations
def transformNoNormalize(width:int=1280, height:int=720, normalizeMean: bool=False)->tuple[T.Compose]:
  """
  Defines the transformation to be applied to the image without normalization
  (used for the DACS method).

  Args:
    width (int): width of the image. Defaults to 1280.
    height (int): height of the image. Defaults to 720.
    normalizeMean (bool): whether to normalize the image or not. Defaults to False.
      If True, the image will be normalized using the mean and std of the ImageNet dataset.

  Returns:
    transformations (tuple[T.Compose]): composed transformation.
      - transformations to the input tensor
      - transformations to the mask tensor
  """
  return T.Compose([
      T.Resize((height, width)), 
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[1,1,1]) if normalizeMean else T.Lambda(lambda x: x)
    ]),  T.Compose([
      T.Resize((height, width),interpolation=T.InterpolationMode.NEAREST),
      T.ToTensor(), 
      T.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])