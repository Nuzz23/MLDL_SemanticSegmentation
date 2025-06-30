import torch, wandb, subprocess, os

from utils import poly_lr_scheduler, BiSeNetLoss, BiSeNetV2Loss
from datasets.dataLoading import loadGTA5, transformationGTA5, loadData, transformationCityScapes
from models.bisenet.build_bisenet import BiSeNet
from Extension.BiSeNetV2.model import BiSeNetV2
from train.trainBiSeNetCity import trainBiSeNet, validateBiSeNet
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation



def init_model(model_str:str=None, totEpoches:int=50, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
            batchSize:int=3, momentum:float=0.9, learning_rate:float=0.005, restartTraining:bool=False, pushWeights:bool=False, enablePrint:bool=False,
            enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None, runId:str|None=None) -> torch.nn.Module:
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
        case 'bisenetv2': model, trainSize = BiSeNetV2(n_classes=19).cuda(), valSize
        case _: model, model_str = BiSeNet(num_classes=19, context_path='resnet18').cuda(), 'bisenet'

    if restartTraining:
        artifact = wandb.use_artifact(f'tempmailforme212-politecnico-di-torino/{model_str}Gta5Aug{augmentation.__str__() if isinstance(augmentation, BaseTransformation) else "+".join(list(map(lambda x:x.__str__(), augmentation)))}/Gta5Aug-weights:latest', type='model')
        artifact_dir = artifact.download()

        # Carica i pesi nel modello
        weights_path = f"{artifact_dir}/Gta5Aug_weights.pth"
        model.load_state_dict(torch.load(weights_path))

        print("Correctly loaded weights from the cloud of WandB!")

        starting_epoch = artifact.metadata['epoch']
    else:
        starting_epoch = 0

    wandb.init(
        project=f'{model_str}Gta5Aug{augmentation.__str__() if isinstance(augmentation, BaseTransformation) else "+".join(list(map(lambda x:x.__str__(), augmentation))) }',
                **({"id":runId, "resume":'must'} if runId else {}),
                config={"starting_epoch": starting_epoch, "epoches":totEpoches, 'weight_decay':1e-4,
                        "learning_rate":learning_rate, "momentum":momentum,'batch_size':batchSize})

    return main(wandb, model=model, model_str=model_str, trainSize=trainSize, valSize=valSize, augmentation=augmentation, pushWeights=pushWeights, 
                enablePrint=enablePrint, enablePrintVal=enablePrintVal, enableProbability=enableProbability)

def main(wandb, model,model_str, trainSize:int=(1280, 720), valSize:int=(1024, 512), augmentation:BaseTransformation|None=None,
        pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False, enableProbability:dict[str, int|float]=None) -> torch.nn.Module:
    """
        Runs the training and validation process of the model.

        Args:
            wandb (wandb): wandb object to log the results
            model (torch.nn.Module): the base model to train.
            trainSize (tuple(int), optional): the size of the training images. Defaults to (1280, 720).
            valSize (tuple(int), optional): the size of the validation images. Defaults to (1024, 512).
            weightPath (str, optional): the path to save the weights. Defaults to 'weights/BiSeNet/'.
            pushWeights (bool, optional): whether to push the weights of the training to git. Defaults to False.
            enablePrint (bool, optional): whether to enable print of the images during training. Defaults to False.
            enablePrintVal (bool, optional): whether to enable print of the images during validation. Defaults to False.
            enableProbability (dict[str, int|float], optional): dictionary to enable the probability of the augmentation. Defaults to None.

        Returns:
            model (torch.nn.Module): the trained model.
    """
    config = wandb.config

    transform_train, transform_groundTruth = transformationCityScapes(width=valSize[0], height=valSize[1])
    _, val_dataloader = loadData(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                                transform_train=transform_train, transform_groundTruth=transform_groundTruth)

    transform_train, transform_groundTruth = transformationGTA5(width=trainSize[0], height=trainSize[1])
    train_dataloader = loadGTA5(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                transform_train=transform_train, transform_groundTruth=transform_groundTruth, augmentation=augmentation, enableProbability=enableProbability)


    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': loss_fn = BiSeNetV2Loss
        case _ : loss_fn = BiSeNetLoss


    for epoch in range(config['starting_epoch'], config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        print(f"\nepoch: {epoch+1:2d} \n\t- Learning Rate -> {lr}")

        train_miou, loss_val = trainBiSeNet(epoch, model, train_dataloader, criterion, loss_fn, optimizer, enablePrint=enablePrint)
        print(f"\t- Train mIoU -> {train_miou}")
        print(f'\t- loss -> {loss_val}')

        val_miou = validateBiSeNet(model, val_dataloader, criterion, enablePrint=enablePrintVal)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, 'loss_val':loss_val, "learning_rate": lr, "epoch":epoch})

        if pushWeights:
            with open("statsCsv/BiSeNetTrainGta5Aug.csv", 'a', encoding='UTF-8') as fp:
                fp.write(f"\n{epoch},{train_miou},{val_miou},{lr},T={enableProbability['T'] if enableProbability and 'T' in enableProbability else None};limit=10,{model_str}")

            try:
                subprocess.run(["git", "add", "statsCsv/BiSeNetTrainGta5Aug.csv"], check=True)
                subprocess.run(["git", "commit", "-m", "added statsCsv/BiSeNetTrainGta5Aug.csv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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