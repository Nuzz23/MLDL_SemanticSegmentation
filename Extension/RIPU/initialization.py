import torch
import wandb

def init_RIPU(model, totEpoches:int=30, gtaSize:tuple[int]=(1024, 512), cityScapesSize:tuple[int]=(1024, 512),
            momentum:float=0.9, learningRate:float=1e-4, weightDecay:float=5e-4, startingLabeledImages:int|float=0.01, 
            epochesToAddLabeledImages:int=3, topK:int|float=20, restartTrainingFromZero:bool=True, enablePrint:bool|int|None=30,
            enablePrintEval:bool|int|None=30, modelIsTrained:bool|str=False):
    
    # load the weight if the model is not already trained
    
    # create the data loaders for the GTA5 and CityScapes datasets
    # create the optmizer and the losses (to rewrite the loss used not to inizialize them again)
    # fix wandb
    # add the training loop
    # add the selection of new images to label
    # add the evaluation function of the model
    # add the printing of the results
    # add the saving of the model
    
    pass