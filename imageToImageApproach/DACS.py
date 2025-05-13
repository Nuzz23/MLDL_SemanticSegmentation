import torch
import numpy as np
from random import sample

def DACS(Xs: torch.Tensor, Ys: torch.Tensor, Xt: torch.Tensor, Yt: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Domain Adaptation using DACS (Domain Adaptation via Class Selection) method.
    This method selects a subset of classes from the source domain and applies them to the target domain.
    
    Args:
        Xs (torch.Tensor): Source domain images.
        Ys (torch.Tensor): Source domain labels.
        Xt (torch.Tensor): Target domain images.
        Yt (torch.Tensor): Target domain labels.
        
    Returns:
        Xt (torch.Tensor): Adapted target domain images.
        Yt (torch.Tensor): Adapted target domain labels.
    """
    Xs, Xt, Ys, Yt = Xs.clone(), Xt.clone(), Ys.clone(), Yt.clone()

    presentClass = [set(map(int, np.unique(image.cpu()))).difference({255}) for image in Ys]
    selectedClass = list(map(lambda x: torch.tensor(sample(list(x), len(x) // 2)), presentClass))
    
    for i in range(len(selectedClass)):
        mask_Ys = torch.isin(Ys[i], selectedClass[i])
        mask_Xs = mask_Ys.expand((3, Ys.shape[2], Ys.shape[3]))
        Xt[i][mask_Xs] = Xs[i][mask_Xs]      
        Yt[i][mask_Ys] = Ys[i][mask_Ys]
        
    return Xt, Yt