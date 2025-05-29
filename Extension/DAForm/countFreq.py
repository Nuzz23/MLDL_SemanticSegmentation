import json
from datasets.dataLoading import transformationGTA5
from datasets.gta5 import GTA5
import torch
import os

def evaluateClassFrequencies(num_classes: int=19):
    """
    Evaluates the class frequencies saving the results in two json files:
    - imagesLabel.json: contains the list of images for each class
    - frequencies.json: contains the frequency of each class
    
    Args:
        num_classes (int): Number of classes in the dataset. Default is 19.
        
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
        
        for label in set(map(lambda x:int(x), torch.unique(mask))).difference({255}):    
                frequencies[label] += 1
                imagesLabel[label].add(batch_idx)
        
    with open("imagesLabel.json", "w") as f:
        json.dump({k: list(v) for k, v in imagesLabel.items()}, f, indent=4)

    with open("frequencies.json", "w") as f:
        json.dump(frequencies, f, indent=4)
    return imagesLabel, frequencies


def loadFrequenciesAndImages(num_classes: int=19) -> tuple:
    """
    Load the class frequencies from the saved JSON files.
    
    Returns:
        Tuple[Dict[int, List[int]], Dict[int, int]]: A tuple containing two dictionaries:
            - imagesLabel: mapping of class labels to the list of image indices
            - frequencies: mapping of class labels to their frequency counts
    """
    pathLabels = "./Extension/DAForm/imagesLabel.json"
    pathFrequencies = "./Extension/DAForm/frequencies.json"

    if not (os.path.exists(pathLabels) and os.path.exists(pathFrequencies)):
        return evaluateClassFrequencies(num_classes=num_classes)

    with open(pathLabels, "r") as f:
        imagesLabel = dict(map(lambda x: (int(x[0]), set(x[1])), json.load(f).items()))

    with open(pathFrequencies, "r") as f:
        frequencies = dict(map(lambda x: (int(x[0]), x[1]), json.load(f).items()))

    return imagesLabel, frequencies
