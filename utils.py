import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# %% Function to compute the learning rate decay

def poly_lr_scheduler(optimizer, init_lr: float, iter: int, lr_decay_iter: float = 1, max_iter: int = 300, power: float = 0.9):
    """Polynomial decay of learning rate

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        init_lr (float): initial learning rate
        iter (int): current iteration
        lr_decay_iter (float, optional): decay iteration. Defaults to 1.
        max_iter (int, optional): maximum iterations. Defaults to 300.
        power (float, optional): power for polynomial decay. Defaults to 0.9.

    Returns:
        learning rate (float): current learning rate
    """
    if (iter % lr_decay_iter == 0) and (iter < max_iter):
      optimizer.param_groups[0]['lr'] = init_lr * (1 - iter / max_iter) ** power

    return optimizer.param_groups[0]['lr']


# %% Function to compute the mean Intersection over Union (IoU) loss

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


def perClassIoU(true,pred, n:int=19):
    """Compute the per-class Intersection over Union (IoU)

    Args:
        true (np.ndarray): the ground truth labels
        pred (np.ndarray): the predicted labels
        n (int, optional): number of classes. Defaults to 19.

    Returns:
        per-class IoU (np.ndarray): IoU for each class of shape (n,)
        List of the classes that are present in the true labels (list)
    """
    # print(per_class_iou(fast_hist(true, pred, n)).shape)
    return per_class_iou(fast_hist(true, pred, n)), [i in true.unique() for i in range(n)]


def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes:int=19, ignore_index:int=255, smooth:float=1e-6):
    """
    Compute the Dice Loss for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Model predictions of shape [B, C, H, W].
        target (torch.Tensor): Ground truth masks of shape [B, H, W].
        num_classes (int, optional): Number of classes in the segmentation task. Defaults to 19.
        ignore_index (int, optional): Value to ignore in the target mask. Defaults to 255.
        
    Returns:
        loss (torch.Tensor): Computed Dice Loss.
    """
    # Crea la rappresentazione one-hot della maschera target
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Ignora i pixel con valore 255
    mask = target != ignore_index
    pred = pred * mask
    target_one_hot = target_one_hot * mask.unsqueeze(1)

    # Calcolo della Dice Loss
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# %% Function to compute the loss for BiSeNet
def BiSeNetLoss(pred: torch.Tensor, mask: torch.Tensor, criterion, weight:float=1) -> torch.Tensor:
    """
    Compute the loss for BiSeNet model.
    
    Args:
        pred (torch.Tensor): Model predictions.
        mask (torch.Tensor): Ground truth masks.
        criterion: Loss function to compute the loss.
        weight (float, optional): Weight for the loss. Defaults to 1.

    Returns:
        loss (torch.Tensor): Computed loss.
    """
    loss1 = criterion(pred[0], mask.long())
    loss2 = criterion(pred[1], mask.long())
    loss3 = criterion(pred[2], mask.long())
    return loss1 + weight*(loss2 + loss3)


def BiSeNetV2Loss(pred: torch.Tensor, mask: torch.Tensor, criterion, weight:float=0.4) -> torch.Tensor:
    """Compute the loss for BiSeNetV2 model.
    
    Args:
        pred (torch.Tensor): Model predictions.
        mask (torch.Tensor): Ground truth masks.
        criterion: Loss function to compute the loss.
        weight (float, optional): Weight for the auxiliary losses. Defaults to 0.4.
    
    Returns:
        loss (torch.Tensor): Computed loss."""
    loss_main = criterion(pred[0], mask.long())
    loss_aux2 = criterion(pred[1], mask.long())
    loss_aux3 = criterion(pred[2], mask.long())
    loss_aux4 = criterion(pred[3], mask.long())
    loss_aux5_4 = criterion(pred[4], mask.long())
    return loss_main + weight * (loss_aux2 + loss_aux3 + loss_aux4 + loss_aux5_4)


def singleCharbonnier(x, nu:int =2):
    """    Compute the Charbonnier loss for a single prediction.
    Args:
        x (torch.Tensor): Model predictions of shape [B, C, H, W].
        nu (int, optional): Exponent for the Charbonnier loss. Defaults to 2.
    Returns:
        charbonnier loss (torch.Tensor): Computed Charbonnier loss."""
    P = F.softmax(x, dim=1)        # [B, 19, H, W]
    logP = F.log_softmax(x, dim=1) # [B, 19, H, W]
    ent = -1.0 * (P * logP).sum(dim=1)  # [B, 1, H, W]
    ent = ent / 2.9444         # change when classes is not 19
    return  ((ent ** 2.0 + 1e-8) ** nu).mean()

def charbonnierEntropy(preds, nu:int = 2):
    """Compute the Charbonnier loss for multiple predictions.
    
    Args:
        preds (list[torch.Tensor]): List of model predictions, each of shape [B, C, H, W].
        nu (int, optional): Exponent for the Charbonnier loss. Defaults to 2.
    
    Returns:
        charbonnier loss (torch.Tensor): Computed Charbonnier loss."""
    x1 = singleCharbonnier(preds[0], nu)
    x2 = singleCharbonnier(preds[1], nu)
    x3 = singleCharbonnier(preds[2], nu)
    return x1 + x2 + x3


# %% Function to visualize the segmentation mask


def print_mask(mask, title:str="title", numClasses:int=19)->None:
    """
        Visualizes the segmentation mask by mapping each class to a specific color.
    
    Args:
        mask (np.ndarray): The segmentation mask to visualize.
        title (str, optional): Title for the plot. Defaults to "title".
        numClasses (int, optional): Number of classes in the segmentation mask. Defaults to 19.
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
    
    
def printHistIou(hist, presentClasses =[range(19)] )->None:
    """
        Visualizes the histograms for Intersection over Union (IoU) calculation.
    
    Args:
        hist vecotr of the percentage for the classes
        presentClasses (list, optional): List of the classes that are present in the true labels. Defaults to [range(19)].
    """
    class_names = [
            'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 
            'Traffic Light', 'Traffic Sign', 'Vegetation', 'Terrain', 'Sky', 
            'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 
            'Motorcycle', 'Bicycle']
    
    # Get labels for x-axis
    if isinstance(presentClasses, list) and len(presentClasses) == len(hist):
        x_labels = [class_names[i] if presentClasses[i] else f"{i}-{class_names[i]}" for i in range(len(hist))]
    else:
        x_labels = [class_names[i] for i in range(len(hist))]
    
    # Plot the histogram
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(hist)), hist, color='blue', alpha=0.7)
    plt.xticks(range(len(hist)), x_labels, rotation=45, ha='right')
    plt.xlabel('Classes')
    plt.ylabel('IoU')
    plt.title('IoU per Class')
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.show()
