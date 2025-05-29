import torch
import json
from datasets.dataLoading import transformationGTA5
from datasets.gta5 import GTA5
import os
import numpy as np

def evaluateClassFrequencies(num_classes: int=19, width:int=1280, height:int=720, useOurs:bool=False, pathLabels:str="./Extension/DAForm/imagesLabel.json", 
                                pathFrequencies:str="./Extension/DAForm/frequencies.json")-> tuple[dict[int, set[int]], dict[int, int]]:
    """
    Evaluates the class frequencies saving the results in two json files:
    - imagesLabel.json: contains the list of images for each class
    - frequencies.json: contains the frequency of each class
    
    Args:
        num_classes (int): Number of classes in the dataset. Default is 19.
        width (int): Width of the images. Default is 1280.
        height (int): Height of the images. Default is 720.
        useOurs (bool): If True, uses our function to evaluate everything else uses the paper method of calculating the frequencies. Default is False.
        pathLabels (str): Path to save the imagesLabel.json file. Default is "./Extension/DAForm/imagesLabel.json".
        pathFrequencies (str): Path to save the frequencies.json file. Default is "./Extension/DAForm/frequencies.json".
        
    Returns:
    Tuple[Dict[int, List[int]], Dict[int, int]]: A tuple containing two dictionaries:
        - imagesLabel: mapping of class labels to the list of image indices
        - frequencies: mapping of class labels to their frequency counts
    """

    imagesLabel, frequencies = {k:set() for k in range(num_classes)}, {k:0 for k in range(num_classes)}
    train, eval = transformationGTA5()

    GTA6 = GTA5('./data/GTA5/', transform=train, transformTarget=eval)

    for batch_idx in range(GTA6.__len__()):    
        mask = GTA6.__getitem__(batch_idx)[1]
        
        for label in map(int, torch.unique(mask[0])):
            if 0 <= label < num_classes:
                imagesLabel[label].add(batch_idx)
                frequencies[label] += mask[0][mask[0]==label].numel()

    if not useOurs:
        frequencies = {k:v/(GTA6.__len__()*width*height) for k,v in frequencies.items()}
        
    with open(pathLabels, "w") as f:
        json.dump({k: list(v) for k, v in imagesLabel.items()}, f, indent=4)

    with open(pathFrequencies, "w") as f:
        json.dump(frequencies, f, indent=4)
        
    return imagesLabel, frequencies


def loadFrequenciesAndImages(num_classes: int=19, width:int = 1280, height:int=720, useOurs:bool=False, 
                                rebuild:bool=False) -> tuple[dict[int, set[int]], dict[int, int]]:
    """
    Load the class frequencies from the saved JSON files.
    
    Args:
        num_classes (int): Number of classes in the dataset. Default is 19.
        width (int): Width of the images. Default is 1280.
        height (int): Height of the images. Default is 720.
        useOurs (bool): If True, uses our function to evaluate everything else uses the paper method of calculating the frequencies. Default is False.
        rebuild (bool): If True, rebuilds the class frequencies even if the JSON files already exist. Default is False.
    
    Returns:
        Tuple[Dict[int, List[int]], Dict[int, int]]: A tuple containing two dictionaries:
            - imagesLabel: mapping of class labels to the list of image indices
            - frequencies: mapping of class labels to their frequency counts
    """
    if useOurs:
        pathLabels = "./Extension/DAForm/imagesLabelOurs.json"
        pathFrequencies = "./Extension/DAForm/frequenciesOurs.json"
    else:
        pathLabels = "./Extension/DAForm/imagesLabel.json"
        pathFrequencies = "./Extension/DAForm/frequencies.json"

    if rebuild or (not (os.path.exists(pathLabels) and os.path.exists(pathFrequencies)) ):
        return evaluateClassFrequencies(num_classes=num_classes, width=width, height=height, 
                                        useOurs=useOurs, pathLabels=pathLabels, pathFrequencies=pathFrequencies) 

    with open(pathLabels, "r") as f:
        imagesLabel = dict(map(lambda x: (int(x[0]), set(x[1])), json.load(f).items()))

    with open(pathFrequencies, "r") as f:
        frequencies = dict(map(lambda x: (int(x[0]), x[1]), json.load(f).items()))

    return imagesLabel, frequencies


def normalizeFrequencies(frequencies: dict[int, float], T:float=0.1)->list[float]:
    """
    Normalize the class frequencies
    The formula used is: 
    P(x) = exp((1 - f(x)) / T) / sum(exp((1 - f(x)) / T))
    
    Args:
        frequencies (dict[int, float]): A dictionary mapping class labels to their frequencies.
        T (float): Temperature parameter for normalization. Default is 0.1.
        
    Returns:
        normalizedFrequencies (list[float]): A list of normalized frequencies for each class.
    """
    bVal = np.exp((1-np.array(list(map(lambda x:x[1], sorted(frequencies.items(), lambda x:x[0]))))/T))
    return bVal/ bVal.sum()