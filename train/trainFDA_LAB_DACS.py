
import torch, os, subprocess, wandb


from train.trainBiSeNetCity import validateBiSeNet

from utils import meanIoULoss, print_mask, charbonnierEntropy, BiSeNetLoss, BiSeNetV2Loss, poly_lr_scheduler
from imageToImageApproach.DACS import DACS
from imageToImageApproach.FDA import FDASourceToTarget
from Extension.LAB.lab import LAB
from datasets.dataLoading import transformationCityScapes, loadData, transformationGTA5, loadGTA5
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation

from Extension.DiceLoss.diceLoss import DiceLoss
from Extension.DiceLoss.diceLossImplementations import OnlyDiceLossBiSeNet, DiceLossAndBiSeNetLoss

from Extension.BiSeNetV2.model import BiSeNetV2
from models.bisenet.build_bisenet import BiSeNet



def init_model(model_str:str=None,  useFDA:bool=True, totEpoches:int=50, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
                batchSize:int=3, momentum:float=0.9, learning_rate:float=0.005,restartTraining:bool=False, pushWeights:bool=False, enablePrint:bool=False,
                enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, diceLossVal:int=0, runId:str=None) -> torch.nn.Module:
    """
    Initializes the model and starts the training process.

    Args:
        model_str (str, optional): The name of the model to use. Defaults to None.
        useFDA (bool, optional): Whether to use FDA for training or LAB. Defaults to True.
        totEpoches (int, optional): The total number of epochs to train the model. Defaults to 50.
        trainSize (tuple(int), optional): The dimensions of the training pictures. Defaults to 1280x720.
        valSize (tuple(int), optional): The dimensions of the validation image. Defaults to 1024x512.
        augmentation (BaseTransformation|None, optional): The augmentation to apply to the training images. Defaults to None.
        batchSize (int, optional): The batch size to use for training. Defaults to 4.
        momentum (float, optional): The momentum to use for the optimizer. Defaults to 0.9.
        learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 0.005.
        restartTraining (bool, optional): Whether to restart the training process from scratch or use the weights
            of the previous training epoch. Defaults to False, ie start from scratch the training.
        pushWeights (bool, optional): Whether to push the weights of the training to git. Defaults to False.
        enablePrint (bool, optional): Whether to enable print of the images during training. Defaults to False.
        enablePrintVal (bool, optional): Whether to enable print of the images during validation. Defaults to False.
        enableProbability (dict[str, int|float], optional): The probability of applying the augmentation. Defaults to None.
        diceLossVal (int, optional): Set to 0 for no dice, set to 1 for dice, set to -1 for dice + cross entropy. Defaults to 0.
        runId (str, optional): The run ID for WandB. If None, a new run will be created.

    Returns:
        model (torch.nn.Module): The fully trained PyTorch model.
    """
    assert torch.cuda.is_available(), "Use cuda (T4 gpu not enabled)"
    wandb.login(key=os.getenv("WANDB_API_KEY", ''))  
    wandb.init()

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': model, trainSize = BiSeNetV2(n_classes=19).cuda(), valSize
        case _: model,model_str = BiSeNet(num_classes=19, context_path='resnet18').cuda(), 'bisenet'

    if restartTraining:
        artifact = wandb.use_artifact(f'tempmailforme212-politecnico-di-torino/{model_str}{"FDA" if useFDA else "LAB"}DACSFixed/Gta5Aug-weights:latest', type='model')
        artifact_dir = artifact.download()

        weights_path = f"{artifact_dir}/Gta5Aug_weights.pth"
        model.load_state_dict(torch.load(weights_path))

        print("Correctly loaded weights from the cloud of WandB!")

        starting_epoch = artifact.metadata['epoch']
    else:
        starting_epoch = 0

    wandb.init(project=f'{model_str}{"FDA" if useFDA else "LAB"}DACSFixed',
                **({"id": runId, "resume":'must'} if runId else {}),
                config={"starting_epoch": starting_epoch, "epoches":totEpoches, 'weight_decay':1e-4,
                        "learning_rate":learning_rate, "momentum":momentum,'batch_size':batchSize})

    return main(wandb, model=model, model_str=model_str, useFDA=useFDA, trainSize=trainSize, valSize=valSize, augmentation=augmentation,
                pushWeights=pushWeights, enablePrint=enablePrint, enablePrintVal=enablePrintVal, enableProbability=enableProbability, diceLossVal=diceLossVal)


def main(wandb, model, model_str, useFDA, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
        pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, diceLossVal:int=0) -> torch.nn.Module:
    """
        Runs the training and validation process of the model.

        Args:
            wandb (wandb): wandb object to log the results
            model (torch.nn.Module): the base model to train.
            trainSize (tuple(int), optional): the size of the training images. Defaults to (1280, 720).
            valSize (tuple(int), optional): the size of the validation images. Defaults to (1024, 512).
            augmentation (BaseTransformation|None, optional): the augmentation to apply to the training images. Defaults to None.
            pushWeights (bool, optional): whether to push the weights of the training to git. Defaults to False.
            enablePrint (bool, optional): whether to enable print of the images during training. Defaults to False.
            enablePrintVal (bool, optional): whether to enable print of the images during validation. Defaults to False.
            enableProbability (dict[str, int|float], optional): the probability of applying the augmentation. Defaults to None.
            diceLossVal (int, optional): set to 0 for no dice, set to 1 for dice, set to -1 for dice + cross entropy. Defaults to 0.

        Returns:
            model (torch.nn.Module): the trained model.
    """
    config = wandb.config

    if diceLossVal: dice = DiceLoss()

    transform_train, transform_groundTruth = transformationCityScapes(width=trainSize[0], height=trainSize[1])
    cityScapes_train, val_dataloader = loadData(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                                transform_train=transform_train, transform_groundTruth=transform_groundTruth)

    transform_train, transform_groundTruth = transformationGTA5(width=trainSize[0], height=trainSize[1])
    trainGTA = loadGTA5(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                transform_train=transform_train, transform_groundTruth=transform_groundTruth, augmentation=augmentation, enableProbability=enableProbability)


    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': loss_model = BiSeNetV2Loss
        case _: loss_model = BiSeNetLoss

    match diceLossVal:
        case 1: loss_fn = lambda pred, truth, _: OnlyDiceLossBiSeNet(pred, truth, dice)
        case -1: loss_fn = lambda pred, truth, criterion: DiceLossAndBiSeNetLoss(pred, truth, dice, loss_model, criterion)
        case _: loss_fn= loss_model


    for epoch in range(config['starting_epoch'], config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        print(f"\nepoch: {epoch+1:2d} \n\t- Learning Rate -> {lr}")

        train_miou, train_loss = trainBiSeNetFDADACS(model, useFDA, trainGTA, cityScapes_train, criterion, loss_fn, optimizer, enablePrint=enablePrint)
        print(f"\t- Train mIoU -> {train_miou}")

        val_miou = validateBiSeNet(model, val_dataloader, criterion, enablePrint=enablePrintVal, normalize=True)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, "learning_rate": lr, "epoch":epoch, "train_loss":train_loss})

        if pushWeights:
            with open("statsCsv/BiSeNetTrainGta5LABDACS.csv", 'a', encoding='UTF-8') as fp:
                fp.write(f"\n{epoch},{train_miou},{val_miou},{lr},{model_str},{'FDA' if useFDA else 'LAB'}")

            try:
                subprocess.run(["git", "add", "statsCsv/BiSeNetTrainGta5LABDACS.csv"], check=True)
                subprocess.run(["git", "commit", "-m", "added statsCsv/BiSeNetTrainGta5LABDACS.csv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "pull", "--rebase"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "push"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Git operation failed: {e}")

            torch.save(model.state_dict(), "Gta5Aug_weights.pth")

            # Crea un artefatto per i pesi del modello
            artifact = wandb.Artifact(
                name=f"Gta5Aug-weights",
                type="model",
                metadata={"epoch": epoch +1}
            )
            artifact.add_file("Gta5Aug_weights.pth")

            wandb.log_artifact(artifact)

            print("Weights saved as artifacts on WandB!")
            
    print(chr(sum(range(ord(min(str(not())))))))
    return model


def trainBiSeNetFDADACS(model, useFDA, trainGTA, trainCityScapes, criterion, loss_fn, optimizer, enablePrint:bool=False, beta:float=0.05)->float:
    """
    Trains the BiSeNet model using FDA and DACS on the GTA5 dataset and CityScapes dataset.
    
    Args:
        model (torch.nn.Module): The BiSeNet model to train.
        useFDA (bool): Whether to use FDA for training or LAB.
        trainGTA (DataLoader): DataLoader for the GTA5 dataset.
        trainCityScapes (DataLoader): DataLoader for the CityScapes dataset.
        criterion (torch.nn.Module): The loss function to use.
        loss_fn (callable): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        enablePrint (bool): Whether to print the training progress.
        beta (float): The beta value for FDA transformation.

    Returns:
        tuple: A tuple containing the mean IoU and the loss value.
    """
    model.train()
    mean,std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    alphaChar, alphaTarget, mIoU = 0.005, 1, []

    interCity = iter(trainCityScapes)
    assert interCity, "Cityscapes dataset is empty"
    
    if not useFDA: lab = LAB()

    for batch_idx, (inputs, mask) in enumerate(trainGTA):
        mask = mask.squeeze().cuda()
        try:
            preds = model(inputs.cuda())
        except ValueError:
            print(f"Raise error at batch id {batch_idx}")
            continue
        lossSource = loss_fn(preds, mask, criterion) 
        
        curr = next(interCity, None)
        if curr is None:
            interCity = iter(trainCityScapes)
            curr = next(interCity, None)

        imageCity, _ = curr
        del curr, preds

        #calculating the FDA
        modified_source = FDASourceToTarget(inputs, imageCity, beta = beta) if useFDA else lab.transform(inputs.clone(), imageCity.clone())
        charEntropy1 = charbonnierEntropy(model((imageCity.cuda() - mean)/std))
        preds = model((modified_source.cuda() - mean)/ std)
        
        del inputs

        curr = next(interCity, None)
        if curr is None:
            interCity = iter(trainCityScapes)
            curr = next(interCity, None)

        imageCity, _ = curr
        del curr

        Yt = model((imageCity.cuda()-mean)/std) # Yt is the pseudo-label
        Yt = [Yt[i].detach().cpu() for i in range(len(Yt))]

        Yt[0] = Yt[0].argmax(1)

        Xm, Ym = DACS(modified_source, mask.cpu(), imageCity, Yt[0])
        YmPred = model((Xm.cuda()-mean)/std)

        Ym = Ym.cpu()

        #calculating the charbonnierEntropy
        loss =  lossSource + alphaChar*(charEntropy1+charbonnierEntropy(model((imageCity.cuda()-mean)/std))) + alphaTarget*loss_fn(YmPred, Ym.cuda(), criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = preds[0].argmax(1)

        mIoU.append(meanIoULoss(mask, pred).item())

        if not batch_idx % 100 and enablePrint:
            print(f'{batch_idx} --> {loss.item()}')
            print(meanIoULoss(mask, pred).item())
            # Printing Ys
            print_mask(pred[0].cpu(),"Pred - Ys")
            print_mask(mask[0].cpu(),"Mask - Ys")
            # Printing Ym
            print_mask(YmPred[0].argmax(1)[0].cpu(),"Pred - Ym")
            print_mask(Ym[0].cpu(),"Mask - Ym")

    return (sum(mIoU)/len(mIoU),loss.item()) if len(mIoU) else (0,loss.item())