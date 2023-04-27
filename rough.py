#!/usr/bin/env python3
import sys

import zipfile
# with zipfile.ZipFile("datasets/generated_ucmerced.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_ucmerced") 

# with zipfile.ZipFile("datasets/generated_sketch.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_sketch") 


# with zipfile.ZipFile("datasets/generated_eurosat.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_eurosat") 

# with zipfile.ZipFile("datasets/generated_ham10000.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_ham10000") 

# with zipfile.ZipFile("datasets/generated_flowers.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_flowers") 

# with zipfile.ZipFile("datasets/generated_modern_office_31.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_modern_office_31") 

# with zipfile.ZipFile("datasets/generated_clipart.zip", 'r') as zip_ref:
#     zip_ref.extractall("generated_ucmerced") 

# with zipfile.ZipFile("datasets/EuroSAT.zip", 'r') as zip_ref:
#     zip_ref.extractall("eurosat") 

# with zipfile.ZipFile("datasets/clipart.zip", 'r') as zip_ref:
#     zip_ref.extractall("clipart") 

# with zipfile.ZipFile("datasets/sketch.zip", 'r') as zip_ref:
#     zip_ref.extractall("sketch") 

with zipfile.ZipFile("datasets/office-31.zip", 'r') as zip_ref:
    zip_ref.extractall("office-31") 


# arr = [3, 4, 1, 0, 6]
# arr = arr[::-1]
# print(arr)


# method: 0                                     # 0 for SimCLR, 1 for DCL, 2 for MYOW, 3 for SUPERVISED

# # distributed training
# nodes: 1
# gpus: 1                                       # I recommend always assigning 1 GPU to 1 node
# workers: 8
# dataset_dir: "./datasets"                     # path to dataset

# # pretrain options
# seed: 42                                      # sacred handles automatic seeding when passed in the config
# base_batch_size: 100                          # the default should be 64. However, should be varied until better performance is achieved
# base_image_size: 64
# base_lr: 0.03                                 # initial learning rate. This should be varied between 3.0e-3 to 3.0e-2 - If you use LARS, then the lr needs to be decayed
# base_optimizer: "SGD"                        # or LARS (experimental) If you use LARS, then the lr needs to be decayed. I noticed that when SGD-MultiStep was used, the training was very slow. Check row 21. It does not reduce the learning rate
# current_epoch: 0
# start_epoch: 0
# base_epochs: 100                              # this needs to be increased to about 800?? The experiment should start at 100 and then  vary it at an interval of 50
# dataset: 0                                    # dataset type. 0 for IMAGENET, 1 for CIFAR10, 2 for QUICKDRAW, 5 for IMAGENET_LITE, 4 for CLIPART
# base_pretrain: True
# momentum: 0.9                                 # momentum of SGD solver
# resume: ""                                    # path to latest checkpoint (default: none)
# global_step: 0
# log_step: 1000

# # target pretraining options
# target_dataset: 0                             # 2 for QUICKDRAW,  3 for SKETCH, and  4 for CLIPART. 7 CHEST-XRAY UCMerced seems to be a bad dataset. ImageNetLite doesn't work as target ds
# target_batch_size: 64                         # ensure that you get a non-fraction when you do len(dataset)/batch_size
# target_epochs: 0                              # this needs to be increased to about 800?? The experiment should start at 100 and then  vary it at an interval of 50
# target_image_size: 64
# target_lr: 1.0e-3                              # we expect the weights from the base pretraining to be good, so we don't want to distort them too quickly and too much.
# target_optimizer: "SGD"                       # or LARS (experimental) If you use LARS, then the lr needs to be decayed
# target_pretrain: False
# target_epoch_num: 40
# pretrain_path_loss_file: "pretrain_path_loss.pkl"

# # arch options
# resnet: "resnet18"
# projection_dim: 128                           # "[...] to project the representation to a 128-dimensional latent space"

# # loss options
# weight_decay: 1.0e-6                          # "optimized using LARS [...] and weight decay of 1.0e−4" try changing to 1.0e-6 to see the performance
# temperature: 0.1                              # see appendix B.7.: Optimal temperature under different batch sizes

# # reload options
# model_path: "save"                            # directory where all checkpoints would be saved
# epoch_num: 40                                 # use to determine the base checkpoint to be used for the AL cycle
# reload: False                                 # indicates whether to start the training from the checkpoint or not

# # finetuning options
# finetune_dataset: 0                           # 1 for CIFAR10, 2 for QUICKDRAW, 3 for SKETCH, 4 for CLIPART, 7 for CHEST-XRAY. Ideally this should be same as the target_dataset, however some datasets don't have the train folder
# finetune: True
# finetune_batch_size: 128                      # 
# finetune_lr: 1.0e-3 #0.1                            # initial learning rate. This should be varied between 1.0e-3 to 1.0e-2 using LR scheduler
# finetune_optimizer: "Adam-DCL"                # if you use LARS, then the lr needs to be decayed
# finetune_image_size: 64                       # this depends on the dataset used
# finetune_epochs: 50                          # this needs to be increased to 100
# finetune_momentum: 0.9                        # momentum of SGD solver
# finetune_weight_decay: 1.0e-6 #5.0e-4                 # "optimized using LARS [...] and weight decay of 1.0e−6"


# # active learning options
# al_finetune_trainer_epochs: 25
# al_epochs: 20                                 # the default should be kept at 10, however due to compute limitations, I would use 3
# al_batches: 10                                # the default should be kept at 10, however due to compute limitations, I would use 20
# al_batch_size: 256                            # I would like to keep the default at 32 or 64, but I made the current value 8 due to the cuda issue I am having. This seems to be the value that didn't produce an error
# al_finetune_batch_size: 128                   # I would like to keep the default at 32 or 64, but I made the current value 8 due to the cuda issue I am having. This seems to be the value that didn't produce an error
# al_trainer_sample_size: 500                  # this specifies the amount of samples to be added to the training pool after each AL iteration
# al_lr: 0.1
# al_optimizer: "SGD-MultiStepV2"
# al_weight_decay: 5.0e-4
# do_al: True
# al_finetune_data_ratio: 1                     # this indicates the amount of the target data at each batch to be used for finetuning to get the topk
# al_method: 0                                 # 0 for least confidence, 1 for entropy, 2 for both
# al_path_loss_file: "al_path_loss.pkl"
# al_pretext_from_pretrain: True               # this enables the pretext task to be finetuned using the weights of the pretrained model

# ml_project: True                              # switches to ML Final Project mode 
# do_al_for_ml_project: True                    # switches between AL and regular classification









# method: 1                                     # 0 for SimCLR, 1 for DCL, 2 for MYOW, 3 for SUPERVISED

# # distributed training
# nodes: 1
# gpus: 1                                       # I recommend always assigning 1 GPU to 1 node
# workers: 8
# dataset_dir: "./datasets"                     # path to dataset

# # pretrain options
# seed: 42                                      # sacred handles automatic seeding when passed in the config
# base_batch_size: 256                          # the default should be 64. However, should be varied until better performance is achieved
# base_image_size: 32
# base_lr: 0.03                                 # initial learning rate. This should be varied between 3.0e-3 to 3.0e-2 - If you use LARS, then the lr needs to be decayed
# base_optimizer: "SGD"                        # or LARS (experimental) If you use LARS, then the lr needs to be decayed. I noticed that when SGD-MultiStep was used, the training was very slow. Check row 21. It does not reduce the learning rate
# current_epoch: 0
# start_epoch: 0
# base_epochs: 200                              # this needs to be increased to about 800?? The experiment should start at 100 and then  vary it at an interval of 50
# dataset: 1                                    # dataset type. 0 for IMAGENET, 1 for CIFAR10, 2 for QUICKDRAW, 5 for IMAGENET_LITE, 4 for CLIPART
# base_pretrain: True
# momentum: 0.9                                 # momentum of SGD solver
# resume: ""                                    # path to latest checkpoint (default: none)
# global_step: 0
# log_step: 1000

# # target pretraining options
# target_dataset: 1                             # 2 for QUICKDRAW,  3 for SKETCH, and  4 for CLIPART. 7 CHEST-XRAY UCMerced seems to be a bad dataset. ImageNetLite doesn't work as target ds
# target_batch_size: 128                         # ensure that you get a non-fraction when you do len(dataset)/batch_size
# target_epochs: 0                              # this needs to be increased to about 800?? The experiment should start at 100 and then  vary it at an interval of 50
# target_image_size: 32
# target_lr: 1.0e-3                              # we expect the weights from the base pretraining to be good, so we don't want to distort them too quickly and too much.
# target_optimizer: "SGD"                       # or LARS (experimental) If you use LARS, then the lr needs to be decayed
# target_pretrain: False
# target_epoch_num: 40
# pretrain_path_loss_file: "pretrain_path_loss.pkl"

# # arch options
# resnet: "resnet18"
# projection_dim: 128                           # "[...] to project the representation to a 128-dimensional latent space"

# # loss options
# weight_decay: 1.0e-6                          # "optimized using LARS [...] and weight decay of 1.0e−4" try changing to 1.0e-6 to see the performance
# temperature: 0.1                              # see appendix B.7.: Optimal temperature under different batch sizes

# # reload options
# model_path: "save"                            # directory where all checkpoints would be saved
# epoch_num: 40                                 # use to determine the base checkpoint to be used for the AL cycle
# reload: False                                 # indicates whether to start the training from the checkpoint or not

# # finetuning options
# finetune_dataset: 0                           # 1 for CIFAR10, 2 for QUICKDRAW, 3 for SKETCH, 4 for CLIPART, 7 for CHEST-XRAY. Ideally this should be same as the target_dataset, however some datasets don't have the train folder
# finetune: True
# finetune_batch_size: 256                      # 
# finetune_lr: 1.0e-3 #0.1                            # initial learning rate. This should be varied between 1.0e-3 to 1.0e-2 using LR scheduler
# finetune_optimizer: "Adam-DCL"                # if you use LARS, then the lr needs to be decayed
# finetune_image_size: 32                       # this depends on the dataset used
# finetune_epochs: 100                          # this needs to be increased to 100
# finetune_momentum: 0.9                        # momentum of SGD solver
# finetune_weight_decay: 1.0e-6 #5.0e-4                 # "optimized using LARS [...] and weight decay of 1.0e−6"


# # active learning options
# al_finetune_trainer_epochs: 25
# al_epochs: 20                                 # the default should be kept at 10, however due to compute limitations, I would use 3
# al_batches: 10                                # the default should be kept at 10, however due to compute limitations, I would use 20
# al_batch_size: 256                            # I would like to keep the default at 32 or 64, but I made the current value 8 due to the cuda issue I am having. This seems to be the value that didn't produce an error
# al_finetune_batch_size: 128                   # I would like to keep the default at 32 or 64, but I made the current value 8 due to the cuda issue I am having. This seems to be the value that didn't produce an error
# al_trainer_sample_size: 3000                  # this specifies the amount of samples to be added to the training pool after each AL iteration
# al_lr: 0.1
# al_optimizer: "SGD-MultiStepV2"
# al_weight_decay: 5.0e-4
# do_al: True
# al_finetune_data_ratio: 1                     # this indicates the amount of the target data at each batch to be used for finetuning to get the topk
# al_method: 0                                 # 0 for least confidence, 1 for entropy, 2 for both
# al_path_loss_file: "al_path_loss.pkl"
# al_pretext_from_pretrain: False               # this enables the pretext task to be finetuned using the weights of the pretrained model

# ml_project: True                              # switches to ML Final Project mode 
# do_al_for_ml_project: True                    # switches between AL and regular classification

