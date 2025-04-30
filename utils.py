import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
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
    if (iter % lr_decay_iter == 0) and (iter < max_iter):
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


def per_class_iou(hist):
    """Compute the Intersection over Union (IoU) for each class
    Args:
        hist (np.ndarray): confusion matrix of shape (n, n)

    Returns:
        iou (np.ndarray): IoU for each class of shape (n,)
    """

    epsilon = 1e-5
    return (torch.diag(hist)) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + epsilon)


def meanIoULoss(true, pred, n:int=19)->torch.Tensor:
    """Compute the mean Intersection over Union (IoU) loss

    Args:
        true (np.ndarray): the ground truth labels
        pred (np.ndarray): the predicted labels
        n (int, optional): number of classes. Defaults to 19.

    Returns:
        mean IoU (float): mean IoU
    """
    # print(per_class_iou(fast_hist(true, pred, n)).shape)

    #The tensor have 0 vals we need to make te mean with the values that differ from 0
    ## tensor([0.8856, 0.3522, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        # 0.0000]

    # return per_class_iou(fast_hist(true, pred, n))#.mean()

    mean_per_class = per_class_iou(fast_hist(true, pred, n))
    return mean_per_class[[i in true.unique() for i in range(n)]].mean()


def dice_loss_from_logits(logits, targets, num_classes:int, smooth:float=1e-6)->torch.Tensor:
    """
    logits: [B, C, H, W] - raw model outputs
    targets: [B, H, W] - class indices (0 to C-1)
    
    Computes the Dice loss between the predicted logits and the target class indices.
    The logits are first converted to probabilities using softmax, and then the Dice loss is computed.
    The loss is averaged over the batch and classes.
    
    Args:
        logits (torch.Tensor): Raw model outputs of shape [B, C, H, W]
        targets (torch.Tensor): Target class indices of shape [B, H, W]
        num_classes (int): Number of classes
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-6.
        
    Returns:
        diceLoss (torch.Tensor): Dice loss value    
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

    # One-hot encode the targets
    targets_onehot = F.one_hot(targets, num_classes)  # [B, H, W, C]
    targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]

    # Compute per-class Dice
    intersection = (probs * targets_onehot).sum(dim=(2, 3))  # [B, C]
    union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))  # [B, C]

    dice = (2. * intersection + smooth) / (union + smooth)  # [B, C]
    return 1 - dice.mean()  # average over batch and classes


def print_mask(mask, title:str="title", numClasses:int=19)->None:
  """
    function that takes as a input a labeled image and print it according to cityscapes
  """
  cm = plt.get_cmap('gist_rainbow')
  colors = [
    (128, 64, 128),  # 0: road
    (244, 35, 232),  # 1: sidewalk
    (70, 70, 70),    # 2: building
    (102, 102, 156), # 3: wall
    (190, 153, 153), # 4: fence
    (153, 153, 153), # 5: pole
    (250, 170, 30),  # 6: traffic light
    (220, 220, 0),   # 7: traffic sign
    (107, 142, 35),  # 8: vegetation
    (152, 251, 152), # 9: terrain
    (70, 130, 180),  # 10: sky
    (220, 20, 60),   # 11: person
    (255, 0, 0),     # 12: rider
    (0, 0, 142),     # 13: car
    (0, 0, 70),      # 14: truck
    (0, 60, 100),    # 15: bus
    (0, 80, 100),    # 16: train
    (0, 0, 230),     # 17: motorcycle
    (119, 11, 32)    # 18: bicycle
  ]

  new_mask = np.zeros((mask.shape[0], mask.shape[1], 3),dtype=np.uint8)
  new_mask[mask == 255] = (0,0,0)
  for i in range (numClasses):
    new_mask[mask == i] = colors[i][:3]
  plt.figure()
  plt.imshow(new_mask)
  plt.title(title)
  plt.show()