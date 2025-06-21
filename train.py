import torch
from utils import meanIoULoss, print_mask, perClassIoU, BiSeNetLoss
from datasets.dataLoading import loadData, transformationCityScapes
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

# %% TRAINING

def trainDeepLabV2(epoch, model, train_loader, criterion, optimizer, enablePrint:bool=False)->float:
    model.train()
    mIoU = []

    for batch_idx, (inputs, mask) in enumerate(train_loader):
        inputs, mask = inputs.cuda() , mask.squeeze().cuda()
        preds = model(inputs)[0]

        loss = criterion(preds, mask.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = preds.argmax(1)

        mIoU.append(meanIoULoss(mask, pred).item())

        if not batch_idx % 100 and enablePrint:
          print(f'{batch_idx} --> {loss.item()}')
          print(meanIoULoss(mask, pred).item())

          print_mask(pred[0].cpu(),"Pred")
          print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0



def trainBiSeNet(epoch, model, train_loader, loss_fn, criterion, optimizer, enablePrint:bool=False)->float:
    model.train()
    mIoU = []

    for batch_idx, (inputs, mask) in enumerate(train_loader):
        inputs, mask = inputs.cuda() , mask.squeeze().cuda()
        preds = model(inputs)

        loss = loss_fn(preds, mask.long(), criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = preds[0].argmax(1)

        mIoU.append(meanIoULoss(mask, pred).item())

        if not batch_idx % 100 and enablePrint:
          print(f'{batch_idx} --> {loss.item()}')
          print(meanIoULoss(mask, pred).item())

          print_mask(pred[0].cpu(),"Pred")
          print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0, loss.item()


# %% VALIDATION

def validateDeepLabV2(model, val_loader, criterion, enablePrint:bool=False)->float:
    model.eval()
    mIoU = []

    with torch.no_grad():
        for batch_idx, (inputs, mask) in enumerate(val_loader):
            inputs, mask = inputs.cuda(),  mask.squeeze().cuda()
            preds = model(inputs)

            preds = preds.argmax(1)

            mIoU.append(meanIoULoss(mask, preds).item())

            if not batch_idx % 100 and enablePrint:
              # print(f'  val batch:{batch_idx} --> {loss.item()}')   # perchè qui manca la loss? e il criterion?
              print("val: ",meanIoULoss(mask, preds).item())

              print_mask(preds[0].cpu(),"Pred")
              print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0


def validateBiSeNet(model, val_loader, criterion, enablePrint:bool=False)->float:
    model.eval()
    mIoU = []

    with torch.no_grad():
        for batch_idx, (inputs, mask) in enumerate(val_loader):
            inputs, mask = inputs.cuda(),  mask.squeeze().cuda()
            preds = model(inputs)

            loss = criterion(preds, mask.long())

            preds = preds.argmax(1)

            mIoU.append(meanIoULoss(mask, preds).item())

            if not batch_idx % 100 and enablePrint:
              print(f'  val batch:{batch_idx} --> {loss.item()}')
              print("val: ",meanIoULoss(mask, preds).item())

              print_mask(preds[0].cpu(),"Pred")
              print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0




# %% LAST EPOCH EVALUATION
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
    mIoU, pci = lastEpochEvaluation(model,  valCityScape if valCityScape is not None else loadData(batch_size=4, num_workers=2, pin_memory=False,
                                                    transform_train=transform_train, transform_groundTruth=transform_groundTruth)[1] , CrossEntropyLoss(ignore_index=255), enablePrint=enablePrint)
    print(100*pci)
    print('final mIoU', 100*mIoU)
    
    return 100*mIoU, 100*pci

def lastEpochEvaluation(model, val_loader, criterion, enablePrint:bool=False)->float:
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
