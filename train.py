import torch
from utils import meanIoULoss, print_mask, dice_loss_from_logits


def trainDeepLabV2(epoch, model, train_loader, criterion, optimizer, enablePrint:bool=False)->float:
    model.train()
    mIoU = []

    for batch_idx, (inputs, mask) in enumerate(train_loader):
        inputs, mask = inputs.cuda() , mask.squeeze().cuda()
        preds = model(inputs)[0]

        loss = criterion(preds, mask.long()) + dice_loss_from_logits(preds, torch.clamp(mask, 0, 18).long(), 19)

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



def trainBiSeNet(epoch, model, train_loader, criterion, optimizer, enablePrint:bool=False)->float:
    model.train()
    mIoU = []

    for batch_idx, (inputs, mask) in enumerate(train_loader):
        inputs, mask = inputs.cuda() , mask.squeeze().cuda()
        preds = model(inputs)

        loss1 = criterion(preds[0], mask.long())
        loss2 = criterion(preds[1], mask.long())
        loss3 = criterion(preds[2], mask.long())

        loss = loss1 + loss2 + loss3 #dice_loss_from_logits(preds[0], torch.clamp(mask, 0, 18).long(), 19) # #

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

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0



def validate():
    pass



def validate(model, val_loader, criterion, enablePrint:bool=False)->float:
    raise NotImplementedError("This function is not implemented yet.")
    
    
    
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




def validate(model, val_loader, criterion, enablePrint:bool=False)->float:
    raise NotImplementedError("This function is not implemented yet.")
    
    model.eval()
    mIoU = []

    with torch.no_grad():
        for batch_idx, (inputs, mask) in enumerate(val_loader):
            inputs, mask = inputs.cuda(),  mask.squeeze().cuda()
            preds = model(inputs)

            loss = criterion(preds, mask.long())+ dice_loss_from_logits(preds, torch.clamp(mask, 0, 18).long(), 19)

            preds = preds.argmax(1)

            mIoU.append(meanIoULoss(mask, preds).item())

            if not batch_idx % 100 and enablePrint:
              print(f'  val batch:{batch_idx} --> {loss.item()}')
              print("val: ",meanIoULoss(mask, preds).item())

              print_mask(preds[0].cpu(),"Pred")
              print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")