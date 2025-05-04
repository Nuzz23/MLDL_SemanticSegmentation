import torch

def convertLabels(data, mode:bool=True)->torch.Tensor:
    """
    Converts the labels of the data to the format required by the model.

    Args:
        data (torch.Tensor): The input data tensor.
        mode (bool, optional): If True, converts the labels from gta V labels to CityScapes labels.
            If False, converts the labels from CityScapes labels to gta V labels. Defaults to True.

    Returns:
        labels (torch.Tensor): The converted labels tensor.
    """

    labels =    {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18} \
                    if mode else \
                {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
                    10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33}

    data[~torch.isin(data, torch.tensor(list(labels.keys()), dtype=torch.uint8, device=data.device))] = 255

    for k,v in labels.items():
        data[data == k] = v

    return data