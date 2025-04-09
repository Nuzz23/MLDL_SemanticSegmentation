import numpy as np
import matplotlib.pyplot as plt
import torch


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9) -> float:
    """Polynomial decay of learning rate
    
    Args:
        optimizer (torch.optim.Optimizer): optimizer
        init_lr (float): initial learning rate
        iter (int): current iteration
        lr_decay_iter (int, optional): decay iteration. Defaults to 1.
        max_iter (int, optional): maximum iterations. Defaults to 300.
        power (float, optional): power for polynomial decay. Defaults to 0.9.
        
    Returns:
        learning rate (float): current learning rate
    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    optimizer.param_groups[0]['lr'] = init_lr*(1 - iter/max_iter)**power
    return optimizer.param_groups[0]['lr']


def fast_hist(true, pred, n:int):
    """Fast histogram for computing confusion matrix

    Args:
        true (np.ndarray): the ground truth labels
        pred (np.ndarray): the predicted labels
        n (int): number of classes

    Returns:
        confusion matrix (np.ndarray): confusion matrix of shape (n, n)
    """
    
    
    k = (true >= 0) & (true < n)
    return torch.bincount(n * true[k].int() + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist:np.ndarray)->np.ndarray:
    """Compute the Intersection over Union (IoU) for each class
    Args:
        hist (np.ndarray): confusion matrix of shape (n, n)
        
    Returns:
        iou (np.ndarray): IoU for each class of shape (n,)
    """
    
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def meanIoULoss(true: np.ndarray, pred: np.ndarray, n: int=19)->float:
    """Compute the mean Intersection over Union (IoU) loss
    
    Args:
        true (np.ndarray): the ground truth labels
        pred (np.ndarray): the predicted labels
        n (int, optional): number of classes. Defaults to 19.
        
    Returns:
        mean IoU (float): mean IoU
    """
    return per_class_iou(fast_hist(true, pred, n)).mean()



def printImage(tensor:torch.Tensor) -> None:
    """ Display the image, color, and mask using matplotlib
    
    Args:
        tensor (torch.Tensor): tensor containing the image, color, and mask
    """
    image_np, color_np, mask_np = tensor[0].numpy().transpose(1, 2, 0), tensor[1].numpy().transpose(1, 2, 0), tensor[2].numpy().transpose(1, 2, 0)


    # Display the image, color, and mask using matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(color_np)
    plt.title('Color')

    plt.subplot(1, 3, 3)
    plt.imshow(mask_np)
    mask_np[mask_np==255].cumsum()
    plt.title('Mask')

    plt.show()
    print(image_np.shape)