# A3: Active Adversarial Alignment for Source-Free Domain Adaptation

The official pytorch implementation of the paper "A3: Active Adversarial Alignment for Source-Free Domain Adaptation"

## 1. Description
The code is structured as follows:
* config: The hyperparamters used in the project are kept in this file for ease of use.
* datasets: This contains the datasets used in the project
* datautils: Contains files for preprocessing the different datasets for the different levels in the adaptation process
* models: This is the main folder which contains:
    1. active_learning: Handles all the dataset distillation process using active learning.
    2. backbones: This contains the custom ResNet50 backbone.
    3. trainers: This contains the trainers and the linear classifier.
    4. utils: The visualization and other utility tools are contained in this folder.


## 2. How to run
1. Ensure you downloaded and extracted the dataset into the datasets (create it if it does not exist) folder.
2. Download the copy the SwAV weights into the save/checkpoints path (create it if it does not exist).
3. Go to the config/config.yaml file and ensure you update all the hyperameters and parameters as required.
4. Run the following command to install the dependencies for the project:

```
pip install -r requirements.txt
```

5. Then simply run 
```
python main.py
```
to start the training process. The initial configuration follows pretrained source - iterative pretraining - linear classification.

## 3. Pretrained models
As stated, we used the SwAV ImageNet pretrained weights as our source model and it can be downloaded from https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar 

## 4. Important notes
Most of the different evaluations were done by modifying the training pipeline from the main.py file.
