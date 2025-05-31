import torch
from dataclasses import dataclass
from Extension.RIPU.labelingFindingMetrics import computeRAandPA, divideImageIntoSquares
from torch.utils.data import Subset

@dataclass
class LabelingTarget:
    """
    Represents a target image for labeling in the CityScapes dataset.
    
    Attributes:
        idx (int): Index of the image in the dataset.
        metric (float): Metric value computed for the image, used to determine its importance for labeling.
    """
    idx: int
    metric: float


def findNewLabelingTargetImageRIPU(model, cityScapesDataLoader, topK:int=10, width:int=1024, height:int=512,
                                l=32)->list[int]:
    """
    Finds the top K images in the CityScapes dataset that are most interesting to learn to improve the model's predictions.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        cityScapesDataLoader (torch.utils.data.DataLoader): DataLoader for the CityScapes dataset.
        topK (int): The number of top images to return. Default is 10.
        width (int): Width of the images. Default is 1024.
        height (int): Height of the images. Default is 512.
        l (int): Size of the squares to divide the image into. Default is 32.
        
    Returns:
        index (set[int]): List of indices of the top K images in the CityScapes dataset.
    """
    
    model.eval()

    if topK <= 0 or topK > len(cityScapesDataLoader.dataset):
        raise ValueError(f"topK must be between 1 and {len(cityScapesDataLoader.dataset)}")

    mask, topK = divideImageIntoSquares(torch.rand(height, width), k=l), [LabelingTarget(None, float('-inf'))]*topK
    
    for cityImage, _, idx in cityScapesDataLoader:
        pred = model(cityImage.cuda())[0][0,:, :, :]
        curr = LabelingTarget(idx, computeRAandPA(pred, mask))

        if curr.metric > topK[-1].metric:
            topK[-1] = curr
            topK = sorted(topK, key=lambda x: x.metric, reverse=True)

    return set(map(lambda x:x.idx, topK))


def modifyDataLoader(model, cityScapesTotal, supervisedIndex:set[int], unsupervisedIndex:set[int], topK:int=10, width:int=1024, height:int=512,
                     l:int=32)->torch.utils.data.DataLoader:
    """
    Modifies the CityScapes DataLoader to include only the top K images that are most interesting to learn.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        cityScapesSupervised (torch.utils.data.DataLoader): DataLoader for the supervised CityScapes dataset.
        cityScapesUnsupervised (torch.utils.data.DataLoader): DataLoader for the unsupervised CityScapes dataset.
        topK (int): The number of top images to return. Default is 10.
        width (int): Width of the images. Default is 1024.
        height (int): Height of the images. Default is 512.
        l (int): Size of the squares to divide the image into. Default is 32.
        
    Returns:
        modifiedDataLoader (torch.utils.data.DataLoader): Modified DataLoader containing only the top K images.
    """
    newIndices = findNewLabelingTargetImageRIPU(model, Subset(cityScapesTotal, unsupervisedIndex), topK, width, height, l)
    supervisedIndex, unsupervisedIndex = supervisedIndex.union(newIndices), unsupervisedIndex.difference(newIndices)
    
    return Subset(cityScapesTotal, supervisedIndex), Subset(cityScapesTotal, unsupervisedIndex), supervisedIndex, unsupervisedIndex 