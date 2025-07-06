import torch, os, wandb, subprocess

from train.trainBiSeNetCity import validateBiSeNet
from utils import meanIoULoss, print_mask, poly_lr_scheduler, BiSeNetLoss, BiSeNetV2Loss
from imageToImageApproach.DACS import DACS
from datasets.dataLoading import transformationCityScapes, loadData, transformationGTA5, loadGTA5
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from models.bisenet.build_bisenet import BiSeNet
from Extension.BiSeNetV2.model import BiSeNetV2

from Extension.DiceLoss.diceLoss import DiceLoss
from Extension.DiceLoss.diceLossImplementations import OnlyDiceLossBiSeNet, DiceLossAndBiSeNetLoss


from Extension.BiSeNetV2.model import BiSeNetV2
def init_model(model_str:str=None, totEpoches:int=50, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
                batchSize:int=3, momentum:float=0.9, learning_rate:float=0.005, restartTraining:bool=False, pushWeights:bool=False, enablePrint:bool=False,
                enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, runId:str|None= None, useDice:int=-1) -> torch.nn.Module:
    """
    Initializes the model and starts the training process.

    Args:
        model_str (str|None, optional): The name of the model to train. If None, the default model will be used. Defaults to None.
        totEpoches (int, optional): The total number of epochs to train the model. Defaults to 50.
        trainSize (tuple(int), optional): The dimesions of the training pictures. Defaults to 1280x720.
        valSize (tuple(int), optional): The dimesions of the validation image. Defaults to 1024x512.
        augmentation (BaseTransformation|None, optional): The augmentation to apply to the training images. Defaults to None.
        batchSize (int, optional): The batch size to use for training. Defaults to 3.
        momentum (float, optional): The momentum to use for the optimizer. Defaults to 0.9.
        learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 0.005.
        restartTraining (bool, optional): Whether to restart the training process from scratch or use the weights
            of the previous training epoch. Defaults to False, ie start from scratch the training.
        pushWeights (bool, optional): Whether to push the weights of the training to git. Defaults to False.
        enablePrint (bool, optional): Whether to enable print of the images during training. Defaults to False.
        enablePrintVal (bool, optional): Whether to enable print of the images during validation. Defaults to False.
        enableProbability (dict[str, int|float], optional): Dictionary to enable the probability of the augmentation. Defaults to None.
        runId (str, optional): The run ID for WandB. If None, a new run will be created.
        useDice (int, optional): Determines the type of loss to use:
            0: No Dice Loss
            1: Only Dice Loss
            -1: Dice Loss + BiSeNet Loss

    Returns:
        model (torch.nn.Module): The fully trained PyTorch model.
    """
    assert torch.cuda.is_available(), "Use cuda (T4 gpu not enabled)"
    wandb.login(key=os.getenv("WANDB_API_KEY", '')) 
    wandb.init()

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': model, trainSize = BiSeNetV2(n_classes=19).cuda(), valSize
        case _: model, model_str = BiSeNet(num_classes=19, context_path='resnet18').cuda(), 'BiSeNet'

    if restartTraining:
        artifact = wandb.use_artifact(f'tempmailforme212-politecnico-di-torino/{model_str}Gta5AugDACS/Gta5Aug-weights:latest', type='model')
        artifact_dir = artifact.download()

        weights_path = f"{artifact_dir}/Gta5Aug_weights.pth"
        model.load_state_dict(torch.load(weights_path))

        print("Correctly loaded weights from the cloud of WandB!")

        starting_epoch = artifact.metadata['epoch']
    else:
        starting_epoch = 0

    wandb.init(project=f'{model_str}Gta5AugDACS',
                **({"id": runId, "resume":'must'} if runId else {}),
                config={"starting_epoch": starting_epoch, "epoches":totEpoches, 'weight_decay':1e-4,
                        "learning_rate":learning_rate, "momentum":momentum,'batch_size':batchSize})
    
    print(chr(sum(range(ord(min(str(not())))))))
    return main(wandb, model=model, model_str=model_str, trainSize=trainSize, valSize=valSize, augmentation=augmentation, pushWeights=pushWeights, enablePrint=enablePrint, 
                enablePrintVal=enablePrintVal, enableProbability=enableProbability, useDice=useDice)



def main(wandb, model, model_str:str=None, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
            pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, useDice:int=-1) -> torch.nn.Module:
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
            useDice (int, optional): Determines the type of loss to use:
                0: No Dice Loss
                1: Only Dice Loss
                -1: Dice Loss + BiSeNet Loss

        Returns:
            model (torch.nn.Module): the trained model.
    """
    config = wandb.config
    
    if useDice: dice = DiceLoss()

    transform_train, transform_groundTruth = transformationCityScapes(width=trainSize[0], height=trainSize[1])
    cityScapes_train, val_dataloader = loadData(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                                transform_train=transform_train, transform_groundTruth=transform_groundTruth)

    transform_train, transform_groundTruth = transformationGTA5(width=trainSize[0], height=trainSize[1])
    trainGTA = loadGTA5(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                transform_train=transform_train, transform_groundTruth=transform_groundTruth, augmentation=augmentation, enableProbability=enableProbability)


    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': loss_model = BiSeNetV2Loss
        case _: loss_model = BiSeNetLoss

    match useDice:
        case 1: loss_fn = lambda pred, truth, _: OnlyDiceLossBiSeNet(pred, truth, dice)
        case -1: loss_fn = lambda pred, truth, criterion: DiceLossAndBiSeNetLoss(pred, truth, dice, loss_model, criterion)
        case _: loss_fn= loss_model

    for epoch in range(config['starting_epoch'], config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        print(f"\nepoch: {epoch+1:2d} \n\t- Learning Rate -> {lr}")

        train_miou, train_loss = trainBiSeNetDACS(model, trainGTA, cityScapes_train, criterion, loss_fn, optimizer, enablePrint=enablePrint)
        print(f"\t- Train mIoU -> {train_miou}")

        val_miou = validateBiSeNet(model, val_dataloader, criterion, enablePrint=enablePrintVal)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, "learning_rate": lr, "epoch":epoch, "train_loss":train_loss})

        if pushWeights:
            with open("statsCsv/BiSeNetTrainGta5DACS_AUG.csv", 'a', encoding='UTF-8') as fp:
                fp.write(f"\n{epoch},{train_miou},{val_miou},{lr},{augmentation.__str__()}, {model_str}")

            try:
                subprocess.run(["git", "add", "statsCsv/BiSeNetTrainGta5DACS.csv"], check=True)
                subprocess.run(["git", "commit", "-m", "added statsCsv/BiSeNetTrainGta5DACS.csv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "pull", "--rebase"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "push"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Git command failed: {e}")

            # Salva i pesi del modello
            torch.save(model.state_dict(), "Gta5Aug_weights.pth")

            # Crea un artefatto per i pesi del modello
            artifact = wandb.Artifact(
                name=f"Gta5Aug-weights",
                type="model",
                metadata={"epoch": epoch +1}
            )
            artifact.add_file("Gta5Aug_weights.pth")

            # Logga l'artefatto su WandB
            wandb.log_artifact(artifact)

            print("Weights saved as artifacts on WandB!")

    return model


def trainBiSeNetDACS(model, trainGTA, trainCityScapes, criterion, loss_fn, optimizer, alpha: float=1, enablePrint:bool=False)->float:
    """
    Train the BiSeNet model using DACS (Domain Adaptation via Cross Domain mixing).
    
    Args:
        model (torch.nn.Module): The BiSeNet model to train.
        trainGTA (DataLoader): DataLoader for the GTA5 dataset.
        trainCityScapes (DataLoader): DataLoader for the CityScapes dataset.
        criterion (torch.nn.Module): The loss function to use.
        loss_fn (callable): The loss function to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        alpha (float, optional): Weight for the target loss. Defaults to 1.
        enablePrint (bool, optional): Whether to print intermediate results. Defaults to False.
    
    Returns:
        tuple[float, float]: The mean Intersection over Union (mIoU) and the training loss.    
    """
    model.train()
    mIoU = []

    interCity = iter(trainCityScapes)
    assert interCity, "Cityscapes dataset is empty"

    for batch_idx, (inputs, mask) in enumerate(trainGTA):
        inputs, mask = inputs.cuda(), mask.squeeze().cuda()
        preds = model(inputs)
        lossSource = loss_fn(preds, mask, criterion)

        curr = next(interCity, None)
        if curr is None:
            interCity = iter(trainCityScapes)
            curr = next(interCity, None)
        imageCity, _ = curr

        Yt = model(imageCity.cuda()) # Yt is the pseudo-label
        Yt = [Yt[i].detach() for i in range(len(Yt))]

        Yt[0] = Yt[0].argmax(1)

        Xm, Ym = DACS(inputs.cpu(), mask.cpu(), imageCity.cpu(), Yt[0].cpu())
        YmPred = model(Xm.cuda())

        lossTarget = loss_fn(YmPred, Ym.cuda(), criterion)

        loss = lossSource + alpha*lossTarget

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


    return sum(mIoU)/len(mIoU) if len(mIoU) else 0, loss.item()