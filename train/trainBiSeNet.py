import torch
from utils import meanIoULoss, print_mask







def validateBiSeNet2(model, val_loader, criterion, enablePrint:bool=False)->float:
    """
    Validate the BiSeNet model on the validation dataset.
    
    Args:
        model (torch.nn.Module): The BiSeNet model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function to use for validation.
        enablePrint (bool): Whether to print progress and masks during validation. Defaults to False.
        
    Returns:
        float: The mean Intersection over Union (mIoU) for the validation dataset.
    """
    model.eval()
    mIoU = []

    with torch.no_grad():
        for batch_idx, (inputs, mask) in enumerate(val_loader):
            inputs, mask = inputs.cuda(),  mask.squeeze().cuda()
            preds = model(inputs)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds

            loss = criterion(preds, mask.long()) 

            preds = preds.argmax(1)

            mIoU.append(meanIoULoss(mask, preds).item())

            if not batch_idx % 100 and enablePrint:
                print(f'  val batch:{batch_idx} --> {loss.item()}')
                print("val: ",meanIoULoss(mask, preds).item())

                print_mask(preds[0].cpu(),"Pred")
                print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0