import torch, subprocess, os, wandb


from utils import meanIoULoss, print_mask, BiSeNetV2Loss, BiSeNetLoss, charbonnierEntropy, poly_lr_scheduler
from imageToImageApproach.FDA import FDASourceToTarget
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation
from datasets.dataLoading import transformNoNormalize, loadGTA5, loadData
from train.trainBiSeNetCity import validateBiSeNet
from models.bisenet.build_bisenet import BiSeNet
from Extension.BiSeNetV2.model import BiSeNetV2



from Extension.BiSeNetV2.model import BiSeNetV2
def init_model(model_str:str=None, totEpoches:int=50, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
                batchSize:int=3, momentum:float=0.9, learning_rate:float=0.005, restartTraining:bool=False, pushWeights:bool=False, enablePrint:bool=False,
                enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, runId:str|None=None, beta:float=0.01) -> torch.nn.Module:
    """
    Initializes the model and starts the training process.

    Args:
        totEpoches (int, optional): The total number of epochs to train the model. Defaults to 50.
        trainSize (tuple(int), optional): The dimesions of the training pictures. Defaults to 1280x720.
        valSize (tuple(int), optional): The dimesions of the validation image. Defaults to 1024x512.
        augmentation (BaseTransformation|None, optional): The augmentation to apply to the training images. Defaults to None.
        batchSize (int, optional): The batch size to use for training. Defaults to 4.
        momentum (float, optional): The momentum to use for the optimizer. Defaults to 0.9.
        learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 0.005.
        restartTraining (bool, optional): Whether to restart the training process from scratch or use the weigths
            of the previous training epoch. Defaults to False, ie start from scratch the training.
        pushWeights (bool, optional): Whether to push the weights of the training to git. Defaults to False.
        enablePrint (bool, optional): Whether to enable print of the images during training. Defaults to False.
        enablePrintVal (bool, optional): Whether to enable print of the images during validation. Defaults to False.
        enableProbability (dict[str, int|float], optional): Dictionary to enable the probability of the augmentation. Defaults to None.
        runId (str, optional): The run ID for WandB. If None, a new run will be created.

    Returns:
        model (torch.nn.Module): The fully trained PyTorch model.
    """
    assert torch.cuda.is_available(), "Use cuda (T4 gpu not enabled)"
    wandb.login(key=os.environ.get('WANDB_API_KEY', ''))
    wandb.init()

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case "bisenetv2": model, trainSize = BiSeNetV2(n_classes=19).cuda(), valSize
        case _: model, model_str = BiSeNet(num_classes=19, context_path='resnet18').cuda(), 'bisenet'


    if restartTraining:
        artifact = wandb.use_artifact(f'tempmailforme212-politecnico-di-torino/{model_str}Gta5AugFDA/Gta5Aug-weights_beta005_hf:latest', type='model')
        artifact_dir = artifact.download()

        # Carica i pesi nel modello
        weights_path = f"{artifact_dir}/Gta5Aug_weights_beta005_hf.pth"
        model.load_state_dict(torch.load(weights_path))

        print("Correctly loaded weights from the cloud of WandB!")

        starting_epoch = artifact.metadata['epoch']
    else:
        starting_epoch = 0

    wandb.init(project=f'{model_str}Gta5AugFDA',
                **({"id": runId, "resume":'must'} if runId else {}),
                config={"starting_epoch": starting_epoch, "epoches":totEpoches, 'weight_decay':1e-4,
                        "learning_rate":learning_rate, "momentum":momentum,'batch_size':batchSize})

    return main(wandb, model=model, model_str=model_str, trainSize=trainSize, valSize=valSize, augmentation=augmentation,
                pushWeights=pushWeights, enablePrint=enablePrint, enablePrintVal=enablePrintVal, enableProbability=enableProbability, beta=beta)



def main(wandb, model, model_str:str=None, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
         pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, beta:float=0.01) -> torch.nn.Module:
    """
        Runs the training and validation process of the model.

        Args:
            wandb (wandb): wandb object to log the results
            model (torch.nn.Module): the base model to train.
            trainSize (tuple(int), optional): the size of the training images. Defaults to (1280, 720).
            valSize (tuple(int), optional): the size of the validation images. Defaults to (1024, 512).
            augmentation (BaseTransformation|None, optional): the augmentation to apply to the training images. Defaults to None.
            weightPath (str, optional): the path to save the weights. Defaults to 'weights/BiSeNet/'.
            pushWeights (bool, optional): whether to push the weights of the training to git. Defaults to False.
            enablePrint (bool, optional): whether to enable print of the images during training. Defaults to False.
            enablePrintVal (bool, optional): whether to enable print of the images during validation. Defaults to False.
            enableProbability (dict[str, int|float], optional): dictionary to enable the probability of the augmentation. Defaults to None.

        Returns:
            model (torch.nn.Module): the trained model.
    """
    config = wandb.config

    transform_train, transform_groundTruth = transformNoNormalize(width=trainSize[0], height=trainSize[1], normalizeMean=False)
    cityScapes_train, val_dataloader = loadData(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                                transform_train=transform_train, transform_groundTruth=transform_groundTruth)

    transform_train, transform_groundTruth = transformNoNormalize(width=trainSize[0], height=trainSize[1], normalizeMean=False)
    trainGTA = loadGTA5(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                transform_train=transform_train, transform_groundTruth=transform_groundTruth, augmentation=augmentation, enableProbability=enableProbability)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case "bisenetv2": loss_fn = BiSeNetV2Loss
        case _: loss_fn = BiSeNetLoss

    for epoch in range(config['starting_epoch'], config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        print(f"\nepoch: {epoch+1:2d} \n\t- Learning Rate -> {lr}")

        train_miou, train_loss = trainBiSeNetFDA(model, trainGTA, cityScapes_train, criterion, loss_fn, optimizer, enablePrint=enablePrint, beta=beta)
        print(f"\t- Train mIoU -> {train_miou}")

        val_miou = validateBiSeNet(model, val_dataloader, criterion, enablePrint=enablePrintVal, normalize=True)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, "learning_rate": lr, "epoch":epoch,"train_loss":train_loss, "beta": beta})

        if pushWeights:
            with open("statsCsv/BiSeNetTrainGta5FDA.csv", 'a', encoding='UTF-8') as fp:
                fp.write(f"\n{epoch},{train_miou},{val_miou},{lr},{model_str}")

            try:
                subprocess.run(["git", "add", "statsCsv/BiSeNetTrainGta5FDA.csv"], check=True)
                subprocess.run(["git", "commit", "-m", "added statsCsv/BiSeNetTrainGta5FDA.csv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                subprocess.run(["git", "pull", "--rebase"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                subprocess.run(["git", "push"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Git command failed: {e}")

            torch.save(model.state_dict(), "Gta5Aug_weights_beta005_hf.pth")

            artifact = wandb.Artifact(
                name=f"Gta5Aug-weights_beta005_hf",
                type="model",
                metadata={"epoch": epoch +1}
            )
            artifact.add_file("Gta5Aug_weights_beta005_hf.pth")

            wandb.log_artifact(artifact)
            print("Weights saved as artifacts on WandB!")

    return model


def trainBiSeNetFDA(model, trainGTA, trainCityScapes, criterion, loss_fn, optimizer, enablePrint:bool=False, beta:float=0.05)->tuple[float, float]:
    """
    Trains the BiSeNet model using the FDA (Fourier Domain Adaptation) method.

    Args:
        model (torch.nn.Module): The BiSeNet model to train.
        trainGTA (DataLoader): DataLoader for the GTA5 dataset.
        trainCityScapes (DataLoader): DataLoader for the CityScapes dataset.
        criterion (torch.nn.Module): Loss function to compute the loss.
        loss_fn (torch.nn.Module): Loss function for the BiSeNet model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        enablePrint (bool, optional): Whether to print intermediate results. Defaults to False.
    beta (float, optional): The beta parameter for the FDA. Defaults to 0.05.
    
    Returns:
        tuple[float, float]: Mean Intersection over Union (mIoU) and loss value
    """
    
    model.train()
    mIoU, alpha = [], 0.005

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    interCity = iter(trainCityScapes)
    assert interCity, "Cityscapes dataset is empty"

    for batch_idx, (inputs, mask) in enumerate(trainGTA):
        inputs, mask = inputs.cuda() , mask.squeeze().cuda()

        curr = next(interCity, None)
        if curr is None:
            interCity = iter(trainCityScapes)
            curr = next(interCity, None)

        imageCity = curr[0].cuda()

        #calculating the FDA
        modified_source = (FDASourceToTarget(inputs, imageCity, beta = beta).cuda()-mean)/std
        preds = model(modified_source)

        targetPreds = model((imageCity-mean)/std)
        loss = loss_fn(preds, mask, criterion) + charbonnierEntropy(targetPreds) * alpha

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

        del modified_source, imageCity
    return (sum(mIoU)/len(mIoU),loss.item()) if len(mIoU) else (0,loss.item())