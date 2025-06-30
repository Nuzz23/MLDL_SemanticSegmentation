# SEMANTIC SEGMENTATION
## Machine Learning and Deep Learning

This folder contains the project developed for the course Machine Learning and Deep Learning aa 2024/2025 @ PoliTO regarding the topic of semantic segmentation.<br>
Our Model can be tested on https://huggingface.co/spaces/Nuzz23/VisualSemSeg

Authors:
- Aldo Karamani
- Tommaso La Malfa
- Nunzio Licalzi
- Simone Licitra


How the repo is organized:
- `adversarialLearning/`: contains the code for the discriminator of the adversarial learning approach.
- `datasets/`: contains the code for loading the datasets and handling anything data related, further divided into sub-folders.
    - `dataLoading/`: contains the code for loading the datasets and applying transformations to the images.
    - `dataAugmentation/`: contains the code for data augmentation techniques.
        - `base/`: contains the base abstract class for data augmentation transformations.
    - `dataProcessing/`: contains the code for data processing and transformation.
    - `cityscapes/`: contains the cityscapes class that extends torch data loader and is used for loading the cityscapes dataset.
    - `gta5/`: contains the gta5 class that extends torch data loader and is used for loading the gta5 dataset.
    - `downloader/`: contains the code for downloading the datasets and the weight of deeplab
    - `labelConverter/`: contains the code for converting the labels of the datasets GTAV to the cityscapes format.
- `Extension/`: contains the code for the extension proposed by our group.
    - `BiSeNetV2/`: contains the code for the BiSeNetV2 model.
    - `DAForm/`: contains the code for the probability adjustment between the two classes.
    - `DiceLoss/`: contains the code for the Dice loss function and its implementation.
    - `LAB/`: contains the code for the LAB image To Image approach.
- `ImageToImageApproach/`: contains the code for the image to image approach chosen, so DACS and FDA.
    - `DACS/`: contains the code for the Domain Adaptation via Cross Sampling mixing DACS approach.
    - `FDA/`: contains the code for the Fourier Domain Adaptation FDA approach.
- `models/`: contains the code for the models used in the project.
    - `bisenet/`: contains the code for the BiSeNet model.
    - `deeplab/`: contains the code for the DeepLabV2 model.
- `runningNotebooks/`: contains the Jupyter notebooks used for running the code and testing the models.
    - `BiSeNetFDA.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model with FDA.
    - `BiSeNet.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model.
    - `BiSeNetDACS.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model with DACS.
    - `BiSeNetFDAorLABDACS.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model with FDA or LAB and DACS.
    - `BiSeNetGtaAdversarial.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model with adversarial learning approach on GTA5 dataset.
    - `BiSeNetGtaAugmented.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model with GTA5 augmented dataset as training set.
    - `BiSeNetLAB.ipynb`: contains the code for running the BiSeNet or BiSeNetV2 model with LAB.
    - `DeepLabV2.ipynb`: contains the code for running the DeepLabV2 model.
- `stats`: contains the statistics of the training of the various models and approaches.
- `train/`: contains the code for training the models.
    - `trainBiSeNetGtaAdversarial.py`: contains the code for training the BiSeNet or BiSeNetV2 model with adversarial learning approach on GTA5 dataset.
    - `trainBiSeNetCity.py`: contains the code for training the BiSeNet or BiSeNetV2 model with training on cityscapes dataset.
    - `trainBiSeNetOnGTAAug.py`: contains the code for training the BiSeNet or BiSeNetV2 model with GTA5 augmented dataset as training set.
    - `trainDACS.py`: contains the code for training the DACS approach.
    - `trainDeepLabV2.py`: contains the code for training the DeepLabV2 model.
    - `trainFDA_LAB_DACS.py`: contains the code for training the FDA or LAB and DACS approach.
    - `trainFDA.py`: contains the code for training the FDA approach.
    - `trainLAB.py`: contains the code for training the LAB image to image approach.
- `stats/`: contains the code used to compute the statistics of the training of the various models and approaches, so perClassIoU, FLOPS, and model size.
- `utils/`: contains the utility functions used in the project, such as the printing functions, some losses functions and the learning rate management.