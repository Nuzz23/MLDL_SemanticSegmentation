# SEMANTIC SEGMENTATION
## Machine Learning and Deep Learning

This folder contains the project developed for the course Machine Learning and Deep Learning aa 2024/2025 @ PoliTO regarding the topic of segmantic segmentation.

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
    - `LAB/`: contains the code for the LAB model.
