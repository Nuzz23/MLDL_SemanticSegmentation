import torch, subprocess, os, wandb
from utils import meanIoULoss, print_mask,  poly_lr_scheduler
from datasets.dataLoading import loadData, transformationCityScapes
from torch.nn import CrossEntropyLoss
from models.deeplabv2.deeplabv2 import get_deeplab_v2


# import the model and its' weights
def initModelDeepLabV2(totEpoches:int=50, width:int=1024, height:int=512,batchSize:int=3, momentum:float=0.9, learning_rate:float=0.005,
                restartTraining:bool=False, pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False, runId:str=None)-> torch.nn.Module:
    """
    Initializes the model and starts the training process.

    Args:
        totEpoches (int): Total number of epochs for training. Default is 50.
        width (int): Width of the input images. Default is 1024.
        height (int): Height of the input images. Default is 512.
        batchSize (int): Batch size for training. Default is 3.
        momentum (float): Momentum for the optimizer. Default is 0.9.
        learning_rate (float): Learning rate for the optimizer. Default is 0.005.
        restartTraining (bool): Whether to restart training from saved weights. Default is False.
        pushWeights (bool): Whether to push the weights to WandB. Default is False.
        enablePrint (bool): Whether to enable printing during training. Default is False.
        enablePrintVal (bool): Whether to enable printing during validation. Default is False.
        runId (str, optional): The run ID for WandB. If None, a new run will be created.
        
    Returns:
        model (torch.nn.Module): The fully trained PyTorch model.
    """
    assert torch.cuda.is_available(), "Use cuda (T4 gpu not enabled)"
    wandb.login(key=os.getenv("WANDB_API_KEY"))  
    wandb.init()
    

    model = get_deeplab_v2(pretrain_model_path='./weights/DeepLabV2/weights_0_0.pth').cuda()
    if os.path.exists('./weights/DeepLabV2/weights_0_0.pth'):
        os.remove('./weights/DeepLabV2/weights_0_0.pth')

    if restartTraining:
        artifact = wandb.use_artifact('tempmailforme212-politecnico-di-torino/DeepLabV2/model-weights:latest', type='model')
        artifact_dir = artifact.download()

        # Carica i pesi nel modello
        weights_path = f"{artifact_dir}/model_weights.pth"
        model.load_state_dict(torch.load(weights_path))

        print("Correctly loaded weights from the cloud of WandB!")

        starting_epoch = artifact.metadata['epoch']
    else:
        starting_epoch = 0

    wandb.init(project='DeepLabV2', 
                **({'resume':'must', "id": runId} if runId is not None else {}),
                config={"starting_epoch": starting_epoch, "epoches":totEpoches,
                                            "learning_rate":learning_rate, "momentum":momentum,'batch_size':batchSize})

    return mainDeepLabV2(wandb, model, width, height, pushWeights, enablePrint, enablePrintVal)


def mainDeepLabV2(wandb, model, width:int=1024, height:int=512, pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False):

    """
        Runs the training and validation process of the model.

        Args:
            wandb (wandb): wandb object to log the results
            model (torch.nn.Module): the base model to train.
            width (int, optional): the width of the input image. Defaults to 1024.
            height (int, optional): the height of the input image. Defaults to 512.
            pushWeights (bool, optional): whether to push the weights of the training to git. Defaults to False.
            enablePrint (bool, optional): whether to enable print of the images during training. Defaults to False.
            enablePrintVal (bool, optional): whether to enable print of the images during validation. Defaults to False.

        Returns:
            model (torch.nn.Module): the trained model.
    """
    config = wandb.config

    transform_train, transform_groundTruth = transformationCityScapes(width=width, height=height)
    train_dataloader, val_dataloader = loadData(batch_size=config['batch_size'], num_workers=2, pin_memory=False,
                                                transform_train=transform_train, transform_groundTruth=transform_groundTruth)

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])


    for epoch in range(config['starting_epoch'], config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        print(f"\nepoch: {epoch:2d} \n\t- Learning Rate -> {lr}")

        train_miou = trainDeepLabV2(model, train_dataloader, criterion, optimizer, enablePrint=enablePrint)
        print(f"\t- Train mIoU -> {train_miou}")

        val_miou = validateDeepLabV2(model, val_dataloader, criterion, enablePrint=enablePrintVal)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, "learning_rate": lr, "epoch":epoch})

        if pushWeights:
            with open("DeepLabV2.csv", 'a', encoding='UTF-8') as fp:
              fp.write(f"\n{epoch},{train_miou},{val_miou},{lr}")

            try:
                subprocess.run(["git", "add", "DeepLabV2.csv"], check=True)
                subprocess.run(["git", "commit", "-m", "added DeepLabv2.csv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                subprocess.run(["git", "pull", "--rebase"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                subprocess.run(["git", "push"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during git operations: {e}")

            # Salva i pesi del modello
            torch.save(model.state_dict(), "model_weights.pth")

            # Crea un artefatto per i pesi del modello
            artifact = wandb.Artifact(
              name=f"model-weights",
              type="model",
              metadata={"epoch": epoch +1}
            )
            artifact.add_file("model_weights.pth")

            # Logga l'artefatto su WandB
            wandb.log_artifact(artifact)

            print("Weights saved as artifacts on WandB!")
    print(chr(sum(range(ord(min(str(not())))))))
    return model


def trainDeepLabV2(model, train_loader, criterion, optimizer, enablePrint:bool=False)->float:
    """Trains the DeepLabV2 model on the training dataset.
    
    Args:
        model (torch.nn.Module): The DeepLabV2 model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        criterion (torch.nn.Module): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        enablePrint (bool, optional): Whether to enable print of the images during training. Defaults to False.
        
    Returns:
        mIoU (float): The mean Intersection over Union (mIoU) of the model on the training dataset.
"""
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


def validateDeepLabV2(model, val_loader, enablePrint:bool=False)->float:
    """Validates the DeepLabV2 model on the validation dataset.
    
    Args:
        model (torch.nn.Module): The DeepLabV2 model to validate.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        enablePrint (bool, optional): Whether to enable print of the images during validation. Defaults to False.
        
    Returns:
        mIoU (float): The mean Intersection over Union (mIoU) of the model on the validation dataset.
    """
    model.eval()
    mIoU = []

    with torch.no_grad():
        for batch_idx, (inputs, mask) in enumerate(val_loader):
            inputs, mask = inputs.cuda(),  mask.squeeze().cuda()
            preds = model(inputs)

            preds = preds.argmax(1)

            mIoU.append(meanIoULoss(mask, preds).item())

            if not batch_idx % 100 and enablePrint:
                print("val: ",meanIoULoss(mask, preds).item())

                print_mask(preds[0].cpu(),"Pred")
                print_mask(mask[0].cpu(),"Mask")

    return sum(mIoU)/len(mIoU) if len(mIoU) else 0
