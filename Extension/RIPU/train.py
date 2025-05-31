from utils import meanIoULoss, print_mask, BiSeNetLoss
import torch
import numpy as np
from Extension.RIPU.loss import LocalConsistentLoss, NegativeLearningLoss


def trainRIPU(model, gtaDataloader, cityScapesDataLoader, optimizer, criterion, alphaLCL:float=0.1, alphaNeg:float=1, enablePrint:bool|int|None=30):
    model.train()
    mIoUGta, mIoUCity = [], []
    
    # the granularity of both datasets is 1 image per batch
    
    gtaDataloader = torch.utils.data.Subset(gtaDataloader, np.random.choice(range(len(gtaDataloader)), len(cityScapesDataLoader)))
        
    for batch_idx, (gtaImage, gtaMask), (cityImage, cityMask)  in enumerate(zip(gtaDataloader, cityScapesDataLoader)):
        gtaImage, gtaMask, cityImage, cityMask = gtaImage.cuda(), gtaMask.cuda(), cityImage.cuda(), cityMask.cuda()
        predsGta = model(gtaImage)

        # losses on the source domain (GTA5)
        loss = BiSeNetLoss(predsGta, gtaMask.long(), criterion)
        loss += alphaLCL*LocalConsistentLoss().forward(predsGta[0])

        # losses on the target domain (CityScapes)
        predsCity = model(cityImage)
        loss += BiSeNetLoss(predsCity, cityMask.long(), criterion)
        loss += alphaNeg*NegativeLearningLoss().forward(predsCity[0].softmax(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predGta = predsGta[0].argmax(1)
        predCity = predsCity[0].argmax(1)

        mIoUGta.append(meanIoULoss(gtaMask, predGta))
        mIoUCity.append(meanIoULoss(cityMask, predCity))

        if enablePrint is not None and enablePrint is not False and not batch_idx%enablePrint:
          print(f'{batch_idx} --> {loss.item()}')
          print(f"mIoUGta = {mIoUGta[-1]:.4f} \t mIoUCity = {mIoUCity[-1]:.4f}")

          print_mask(predGta.cpu(),"PredGta")
          print_mask(gtaMask.cpu(),"MaskGta")
          print_mask(predCity.cpu(),"PredCity")
          print_mask(cityMask.cpu(),"MaskCity")

    return sum(mIoUGta)/len(mIoUGta) if len(mIoUGta) else 0, sum(mIoUCity)/len(mIoUCity) if len(mIoUCity) else 0
