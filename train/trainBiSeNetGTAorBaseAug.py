import os
import torch, wandb
from utils import meanIoULoss, print_mask, poly_lr_scheduler, BiSeNetLoss, BiSeNetV2Loss
from models.bisenet.build_bisenet import BiSeNet
from Extension.BiSeNetV2.model import BiSeNetV2
from datasets.dataLoading import loadData, transformationCityScapes




def initBiSeNetOrV2Base(model_str:str='bisenet', totEpoches:int=50, width:int=1024, height:int=512, batchSize:int=4, momentum:float=0.9, learning_rate:float=0.0005,
                restartTraining:bool=False, pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False):
    """
    Initializes the model and starts the training process.

    Args:
        totEpoches (int, optional): The total number of epochs to train the model. Defaults to 50.
        width (int, optional): The width of the input image. Defaults to 1024.
        height (int, optional): The height of the input image. Defaults to 512.
        batchSize (int, optional): The batch size to use for training. Defaults to 4.
        momentum (float, optional): The momentum to use for the optimizer. Defaults to 0.9.
        learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 0.005.
        restartTraining (bool, optional): Whether to restart the training process from scratch or use the weights
            of the previous training epoch. Defaults to False, ie start from scratch the training.
        pushWeights (bool, optional): Whether to push the weights of the training to git. Defaults to False.
        enablePrint (bool, optional): Whether to enable print of the images during training. Defaults to False.
        enablePrintVal (bool, optional): Whether to enable print of the images during validation. Defaults to False.

    Returns:
        model (torch.nn.Module): The fully trained PyTorch model.
    """
    assert torch.cuda.is_available(), "Use cuda (T4 gpu not enabled)"
    wandb.login(key=os.environ.get('WANDB_API_KEY', ''))
    wandb.init()

    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': model = BiSeNetV2(n_classes=19).cuda()
        case _: model, model_str = BiSeNet(num_classes=19, context_path='resnet18').cuda(), 'bisenet'

    if restartTraining:
        artifact = wandb.use_artifact(f'tempmailforme212-politecnico-di-torino/BiSeNetV2/model-weights:latest', type='model')
        artifact_dir = artifact.download()

        # Carica i pesi nel modello
        weights_path = f"{artifact_dir}/model_weights.pth"
        model.load_state_dict(torch.load(weights_path))

        print("Correctly loaded weights from the cloud of WandB!")

        starting_epoch = artifact.metadata['epoch']
    else:
        starting_epoch = 0

    wandb.init(project=f'BiSeNetV2',
                config={"starting_epoch": starting_epoch, "epoches":totEpoches, 'weight_decay':1e-4,
                        "learning_rate":learning_rate, "momentum":momentum,'batch_size':batchSize})

    return mainBiSeNetBaseCity(wandb, model, model_str, width, height, pushWeights, enablePrint, enablePrintVal)




def mainBiSeNetBaseCity(wandb, model, model_str, width:int=1024, height:int=512, pushWeights:bool=False, enablePrint:bool=False, enablePrintVal:bool=False):
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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])


    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': loss_fn = BiSeNetV2Loss
        case _: loss_fn = BiSeNetLoss

    for epoch in range(config['starting_epoch']-1, config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate'], iter=epoch*len(train_dataloader), max_iter=config['epoches']*len(train_dataloader), lr_decay_iter=1)
        print(f"\nepoch: {epoch+1:2d} \n\t- Learning Rate -> {lr}")

        train_miou, train_loss = trainBiSeNet(model, train_dataloader, criterion, loss_fn, optimizer, enablePrint=enablePrint)
        print(f"\t- Train mIoU -> {train_miou*100:.4f}")
        print(f"\t- Train Loss -> {train_loss:.4f}")

        val_miou = validateBiSeNet(model, val_dataloader, criterion, enablePrint=enablePrintVal)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, "learning_rate": lr, "epoch":epoch,
                   'train_loss':train_loss})

        if pushWeights:
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
    return model


def trainBiSeNet(model, train_loader, criterion, loss_fn, optimizer, enablePrint:bool=False)->float:
    """
    Train the BiSeNet model on the training dataset.
    
    Args:
        model (torch.nn.Module): The BiSeNet model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): The loss function to use for training.
        loss_fn (callable): The loss function specific to the BiSeNet model.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        enablePrint (bool): Whether to print progress and masks during training. Defaults to False.
        
    Returns:
        tuple[float, float]: The mean Intersection over Union (mIoU) and the training loss.
    """
    model.train()
    mIoU = []

    for batch_idx, (inputs, mask) in enumerate(train_loader):
        inputs, mask = inputs.cuda() , mask.squeeze().cuda()
        preds = model(inputs)

        loss = loss_fn(preds, mask, criterion)

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



def validateBiSeNet(model, val_loader, criterion, enablePrint:bool=False)->float:
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