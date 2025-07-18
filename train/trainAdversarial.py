import os, torch, subprocess, wandb

from models.bisenet.build_bisenet import BiSeNet
from Extension.BiSeNetV2.model import BiSeNetV2
from adversarialLearning.discriminator import FCDiscriminator

from datasets.dataLoading import transformationCityScapes, loadData, transformationGTA5, loadGTA5
from datasets.dataAugmentation.base.baseTransformation import BaseTransformation

from utils import meanIoULoss, print_mask, poly_lr_scheduler, BiSeNetLoss, BiSeNetV2Loss
from train.trainBiSeNetCity import validateBiSeNet

from Extension.DiceLoss.diceLoss import DiceLoss
from Extension.DiceLoss.diceLossImplementations import OnlyDiceLossBiSeNet, DiceLossAndBiSeNetLoss

def init_model(model_str, totEpoches: int = 50,trainSize: tuple = (1280, 720),valSize: tuple = (1024, 512),
               augmentation: BaseTransformation | None = None, batchSize: int = 3, momentum: float = 0.9,
               learning_rate_segment: float = 0.05, learning_rate_disc: float = 0.005, restartTraining: bool = False, pushWeights: bool = False,
               enablePrint: bool = False, enablePrintVal: bool = False, runId: str | None = None, useDice:int = -1,
               enableProb:dict|None= None):
    """
    Initializes the model and starts the training process.

    Args:
        totEpoches (int, optional): Total number of epochs for training. Default is 50.
        trainSize (tuple, optional): Training image dimensions (e.g., (1280, 720)).
        valSize (tuple, optional): Validation image dimensions (e.g., (1024, 512)).
        augmentation (BaseTransformation|None, optional): Data augmentation to apply, if any.
        batchSize (int, optional): Batch size for training. Default is 3.
        momentum (float, optional): Momentum for the optimizer. Default is 0.9.
        learning_rate_segment (float, optional): Learning rate for the segmentation model. Default is 0.05.
        learning_rate_disc (float, optional): Learning rate for the discriminator. Default is 0.005.
        restartTraining (bool, optional): If True, loads previously saved checkpoints from WandB.
        pushWeights (bool, optional): If True, checkpoints will be saved to WandB during training.
        enablePrint (bool, optional): If True, enables logging during training.
        enablePrintVal (bool, optional): If True, enables logging during validation.
        runId (str|None, optional): If provided, resumes a specific WandB run.
        useDice (int, optional): Determines the type of loss to use:
            0: No Dice Loss
            1: Only Dice Loss
            -1: Dice Loss + BiSeNet Loss
        enableProb (dict|None, optional): Dictionary to control the probability of enabling augmentation.

    Returns:
        model, discriminator (torch.nn.Module): The fully trained model and discriminator.
    """
    assert torch.cuda.is_available(), "use cuda (T4 gpu not enabled)"

    wandb.login(key=os.environ.get('WANDB_API_KEY', ''))
    wandb.init()

    
    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': model, trainSize = BiSeNetV2(n_classes=19).cuda(), valSize
        case _ : model,model_str = BiSeNet(num_classes=19, context_path='resnet18').cuda(), "BiSeNet"
    discriminator = FCDiscriminator(num_classes=19, ndf=64).cuda()

    if restartTraining:
        artifact = wandb.use_artifact(f'tempmailforme212-politecnico-di-torino/{model_str}Gta5AdversialDICE/Gta5Adv-weights:latest', type='model')
        artifact_dir = artifact.download()

        checkpoint_path = f"{artifact_dir}/Gta5Adv_checkpoint.pth"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        starting_epoch = checkpoint['epoch']+1
    else:
        starting_epoch = 0

    wandb.init(project=f'{model_str}Gta5AdversialDICE',
              **({"id": runId, "resume": 'must'} if runId else {}),
                config={"starting_epoch": starting_epoch, "epoches": totEpoches, 'weight_decay': 1e-4, "learning_rate_segment": learning_rate_segment,
                "learning_rate_disc": learning_rate_disc, "momentum": momentum,'batch_size': batchSize})

    return main(wandb, model, model_str, discriminator, trainSize, valSize, augmentation,
                pushWeights, enablePrint, enablePrintVal, useDice, enableProb), discriminator



def main(wandb, model, model_str, discriminator, trainSize: tuple = (1280, 720), valSize: tuple = (1024, 512),
        augmentation: BaseTransformation= None, pushWeights: bool = False, enablePrint: bool = False, enablePrintVal: bool = False, diceLossVal:int =-1, enableProbability:dict = {'T':0.20, 'limit':4}):
    """
    Main function to train the model using adversarial training with a discriminator.
    
    Args:
        wandb (wandb): WandB object to log the results.
        model (torch.nn.Module): The segmentation model to train.
        model_str (str): The string identifier for the model.
        discriminator (torch.nn.Module): The discriminator model.
        trainSize (tuple, optional): The size of the training images. Defaults to (1280, 720).
        valSize (tuple, optional): The size of the validation images. Defaults to (1024, 512).
        augmentation (BaseTransformation, optional): The data augmentation transformation to apply. Defaults to None.
        pushWeights (bool, optional): Whether to push the weights of the training to git. Defaults to False.
        enablePrint (bool, optional): Whether to enable print of the images during training. Defaults to False.
        enablePrintVal (bool, optional): Whether to enable print of the images during validation. Defaults to False.
        diceLossVal (int, optional): The value to determine the type of loss to use, if 0 no dice loss, 1 for solely dice, -1 for dice and cross entropy. Defaults to -1.
        enableProbability (dict, optional): Dictionary to control the probability of enabling augmentation. Defaults to {'T':0.20, 'limit':4}.
    """
    config = wandb.config

    if diceLossVal: dice = DiceLoss()

    transform_train, transform_groundTruth = transformationCityScapes(width=valSize[0], height=valSize[1])
    trainCityScapes, val_dataloader = loadData(batch_size=config['batch_size'], num_workers=2,pin_memory=False,
                                transform_train=transform_train,transform_groundTruth=transform_groundTruth)

    transform_train, transform_groundTruth = transformationGTA5(width=trainSize[0], height=trainSize[1])
    trainGTA = loadGTA5(batch_size=config['batch_size'],num_workers=2,pin_memory=False,
        transform_train=transform_train,transform_groundTruth=transform_groundTruth, augmentation=augmentation, enableProbability=enableProbability)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(),lr=config['learning_rate_segment'],
                                momentum=config['momentum'],weight_decay=config['weight_decay'])

    criterion_discriminator = torch.nn.MSELoss()
    optimizer_discriminator = torch.optim.SGD(discriminator.parameters(),lr=config['learning_rate_disc'],
                                                momentum=config['momentum']+0.09,weight_decay=config['weight_decay'])


    match model_str.lower() if isinstance(model_str, str) else model_str:
        case 'bisenetv2': loss_model = BiSeNetV2Loss
        case _: loss_model = BiSeNetLoss

    match diceLossVal:
        case 1: loss_fn = lambda pred, truth, _: OnlyDiceLossBiSeNet(pred, truth, dice)
        case -1: loss_fn = lambda pred, truth, criterion: DiceLossAndBiSeNetLoss(pred, truth, dice, loss_model, criterion)
        case _: loss_fn= loss_model

    for epoch in range(config['starting_epoch'], config['epoches']):
        lr = poly_lr_scheduler(optimizer, init_lr=config['learning_rate_segment'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        lrDiscriminator = poly_lr_scheduler(optimizer_discriminator, init_lr=config['learning_rate_disc'], iter=epoch, max_iter=config['epoches'], lr_decay_iter=1)
        print(f"\nepoch: {epoch+1:2d} \n\t- Learning Rate segment-> {lr} \n\t- Learning Rate discriminator-> {lrDiscriminator}")
        wandb.config['starting_epoch'] = epoch

        train_miou, train_loss = adversarial_train(model, discriminator, criterion, loss_fn, criterion_discriminator,
                                                    optimizer, optimizer_discriminator, trainGTA,trainCityScapes, enablePrint=enablePrint)

        print(f"\t- Train mIoU -> {train_miou}\n\t- Train loss -> {train_loss}")

        val_miou = validateBiSeNet(model, val_dataloader, criterion, enablePrint=enablePrintVal)
        print(f"\t- Validate mIoU -> {val_miou}")

        wandb.log({"train_mIoU": train_miou, "val_mIoU": val_miou, "learning_rate_segment": lr, "learning_rate_discriminator": lrDiscriminator, "epoch": epoch, "train_loss":train_loss})

        if pushWeights:
            with open(f"statsCsv/{model_str}Adversarial.csv", 'a', encoding='UTF-8') as fp:
                fp.write(f"\n{epoch},{train_miou},{val_miou},{lr},{lrDiscriminator}")

                try:
                    subprocess.run(["git", "add", f"statsCsv/{model_str}Adversarial.csv"], check=True)
                    subprocess.run(["git", "commit", "-m", f"added statsCsv/{model_str}Adversarial.csv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["git", "pull", "--rebase"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["git", "push"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError as e:
                    print(f"Error during git operations: {e}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict()
            }
            torch.save(checkpoint, "Gta5Adv_checkpoint.pth")

            artifact = wandb.Artifact(
                name=f"Gta5Adv-weights",
                type="model",
                metadata={"epoch": epoch + 1}
            )
            artifact.add_file("Gta5Adv_checkpoint.pth")
            wandb.log_artifact(artifact)

    
            print("Weights saved as artifacts on WandB!")
            
    print(chr(sum(range(ord(min(str(not())))))))
    return model


def adversarial_train(model, discriminator, criterion, loss_fn, loss_discriminator, optimizer, optimizer_discriminator,
                           trainGTA, trainCityScapes, enablePrint: bool = False):
    """    Train the model using adversarial training with a discriminator.
    Args:
        model (torch.nn.Module): The segmentation model to train.
        discriminator (torch.nn.Module): The discriminator model.
        criterion (torch.nn.Module): The loss function for segmentation.
        loss_fn (callable): The function to compute the segmentation loss.
        loss_discriminator (callable): The function to compute the discriminator loss.
        optimizer (torch.optim.Optimizer): The optimizer for the segmentation model.
        optimizer_discriminator (torch.optim.Optimizer): The optimizer for the discriminator.
        trainGTA (torch.utils.data.DataLoader): Dataloader for the source domain (GTA).
        trainCityScapes (torch.utils.data.DataLoader): Dataloader for the target domain (Cityscapes).
        enablePrint (bool): Whether to print progress and masks during training. Defaults to False.
        
    Returns:
        tuple: Average mean IoU and total loss for the epoch.
    """
    
    lambda_adv, mIoU = 0.001, []

    model.train(), discriminator.train()

    interCity = iter(trainCityScapes)
    assert interCity, "Cityscapes dataset is empty"

    for batch_idx, (inputs, mask) in enumerate(trainGTA):
        curr = next(interCity, None)
        if curr is None:
            interCity = iter(trainCityScapes)
            curr = next(interCity, None)
        imageCity = curr[0].cuda()

        # === TRAIN GENERATOR ===
        optimizer.zero_grad() 
        optimizer_discriminator.zero_grad()

        # Freeze discriminator parameters for generator update
        for param in discriminator.parameters():
            param.requires_grad = False

        inputs, mask = inputs.cuda(), mask.squeeze().cuda()
        preds = model(inputs)
        segmentation_loss = loss_fn(preds, mask, criterion)

        output_target = model(imageCity)[0]
        discriminator_output_target = discriminator(torch.nn.functional.softmax(output_target, dim=1))
        discriminator_label_source = torch.ones_like(discriminator_output_target).cuda()

        loss = segmentation_loss + lambda_adv * loss_discriminator(discriminator_output_target, discriminator_label_source)
        loss.backward()
        optimizer.step()

        # === TRAIN DISCRIMINATOR ===
        for param in discriminator.parameters():
            param.requires_grad = True

        discriminator_output_source = discriminator(torch.nn.functional.softmax(preds[0].detach(), dim=1))
        discriminator_label_source = torch.zeros_like(discriminator_output_source).cuda()
        discriminator_loss_source = loss_discriminator(discriminator_output_source, discriminator_label_source)

        discriminator_output_target = discriminator(torch.nn.functional.softmax(output_target.detach(), dim=1))
        discriminator_label_target = torch.ones_like(discriminator_output_target).cuda()
        discriminator_loss_target = loss_discriminator(discriminator_output_target, discriminator_label_target)

        loss2 = discriminator_loss_source + discriminator_loss_target
        loss2.backward()
        optimizer.step()
        optimizer_discriminator.step()

        prediction_source = preds[0].argmax(1)
        # Compute mean IoU for monitoring (using the main output)
        mIoU.append(meanIoULoss(mask, prediction_source).item())

        # Optionally print progress and masks every 100 batches
        if not batch_idx % 100 and enablePrint:
            print(f'  val batch:{batch_idx} --> {loss.item()}')
            print("val: ", meanIoULoss(mask, prediction_source).item())
            print_mask(prediction_source[0].cpu(),"Pred")
            print_mask(mask[0].cpu(),"Mask")

    # Return average mIoU and total loss for the epoch
    return sum(mIoU)/len(mIoU) if len(mIoU) else 0, loss.item()
