import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numpy as np
from time import time as getCurrentTime
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets.dataLoading import transformationCityScapes, loadData
from utils import perClassIoU, meanIoULoss, print_mask


def countFLOPS(model, width:int=1024, height:int=512)->int:
  """
    Counts the number of Floating Point Operations (FLOPs) in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count FLOPs for.
        width (int, optional): The width of the input image. Defaults to 1024.
        height (int, optional): The height of the input image. Defaults to 512.

    Returns:
        FLOPs (int): The total number of FLOPs in the model.

  """
  image = torch.zeros((1, 3, height, width)).cuda()

  return FlopCountAnalysis(model, image).total()


def latency(model, num_iterations:int=1000, width:int=1024, height:int=512)->list[float]:
  """
  Evaluates the latency and fps of a PyTorch model for a given number of iterations.

  Args:
      model (torch.nn.Module): The PyTorch model to evaluate.
      num_iterations (int, optional): The number of iterations to evaluate. Defaults to 1000.
      width (int, optional): The width of the input image. Defaults to 1024.
      height (int, optional): The height of the input image. Defaults to 512.

  Returns:
      latency and FPS (list[float]): The list of values
        - the mean latency of the model
        - the standard deviation of the latency of the model
        - the mean fps of the model
        - the standard deviation of the fps of the model
  """
  image = torch.randn(1, 3, height, width).cuda()

  latency, FPS = np.zeros(num_iterations), np.zeros(num_iterations)

  for i in range(num_iterations):
      latency[i] = getCurrentTime()

      _ = model(image)

      latency[i] = getCurrentTime() - latency[i]
      FPS[i] = 1/latency[i]

  return np.mean(latency)*1000, np.std(latency)*1000, np.mean(FPS), np.std(FPS)



def evaluateLastEpoch(model, valCityScape:DataLoader=None, width:int=1024, height:int=512, enablePrint:bool=False)->tuple[float, float]:
    """
    Evaluates the last epoch of the model on the CityScapes dataset.
    Args:
        model (torch.nn.Module): The model to evaluate.
        width (int): Width of the images. Defaults to 1024.
        height (int): Height of the images. Defaults to 512.
        
    Returns:
        tuple[float, float]: Mean Intersection over Union (mIoU) and per-class IoU (pci).
    """
    transform_train, transform_groundTruth = transformationCityScapes(width=width, height=height)
    if valCityScape is None:
        valCityScape = loadData(batch_size=4, num_workers=2, pin_memory=False,
                                transform_train=transform_train, transform_groundTruth=transform_groundTruth)[1]
    
    mIoU, pci = lastEpochEvaluation(model,  valCityScape, CrossEntropyLoss(ignore_index=255), enablePrint=enablePrint)
    if enablePrint: print(100*pci, '\nfinal mIoU', 100*mIoU)
    return 100*mIoU, 100*pci


def lastEpochEvaluation(model, val_loader, criterion, enablePrint:bool=False)->float:
    """
    Evaluates the model on the validation set and computes the mean Intersection over Union (mIoU) and per-class IoU (pci).
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function to compute the loss.
        enablePrint (bool, optional): Whether to print intermediate results. Defaults to False.
      
    Returns:
        tuple[float, torch.Tensor]: Mean Intersection over Union (mIoU) and per-class IoU (pci).
    """

    model.eval()
    mIoU, IoU = [], torch.zeros(19)

    with torch.no_grad():
        for batch_idx, (inputs, mask) in enumerate(val_loader):
            inputs, mask = inputs.cuda(),  mask.squeeze().cuda()
            preds = model(inputs)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds

            loss = criterion(preds, mask.long())#+ dice_loss_from_logits(preds, torch.clamp(mask, 0, 18).long(), 19)

            preds = preds.argmax(1)
            
            pci, present = perClassIoU(mask, preds)
            pci, present = pci.cpu(), present
            IoU[present] += pci[present]

            mIoU.append(pci[present].mean().item())

            if not batch_idx % 100 and enablePrint:
              print(f'  val batch:{batch_idx} --> {loss.item()}')
              print("val: ",meanIoULoss(mask, preds).item())

              print_mask(preds[0].cpu(),"Pred")
              print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0, IoU/len(mIoU) if len(mIoU) else 0
