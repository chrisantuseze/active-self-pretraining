import torch.nn as nn
import torch
import torchvision
from torch.utils.data import random_split
import gc
from datautils.dataset_enum import DatasetType

from models.heads.nt_xent import NT_Xent
from models.self_sup.simclr.loss.dcl_loss import DCL
from models.self_sup.simclr.loss.nt_xent_loss import NTXentLoss
from models.self_sup.simclr.simclr import SimCLR
from models.self_sup.simclr.simclr_v2 import SimCLRV2
from models.utils.ssl_method_enum import SSL_Method
from models.utils.training_type_enum import Params, TrainingType


def get_model_criterion(args, encoder, training_type=TrainingType.ACTIVE_LEARNING, num_classes=4):
    n_features = get_feature_dimensions_backbone(args)

    if training_type == TrainingType.ACTIVE_LEARNING or training_type == TrainingType.FINETUNING:
        criterion = nn.CrossEntropyLoss()
        model = encoder
        model.linear = nn.Linear(n_features, num_classes)
        print("using Regular model")

    else:
        if args.method == SSL_Method.SIMCLR.value:
            # criterion = NT_Xent(batch_size, args.temperature, args.world_size)
            criterion = NTXentLoss(args)
            model = SimCLR(encoder, args.projection_dim, n_features)
            print("using SIMCLR")

        elif args.method == SSL_Method.DCL.value:
            criterion = DCL(args)
            model = SimCLRV2(n_features)
            print("using SIMCLRv2")

        elif args.method == SSL_Method.SUPERVISED.value:
            criterion = nn.CrossEntropyLoss()
            model = encoder
            print("using Supervised model")
    

    return model, criterion

def get_feature_dimensions_backbone(args):
    if args.resnet == 'resnet18':
        return 512

    elif args.resnet == 'resnet50':
        return 2048

    else:
        raise NotImplementedError

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def get_params_to_update(model, feature_extract):
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    return params_to_update

def get_params(args, training_type):
    # if args.dataset == DatasetType.CIFAR10.value:
    #     base_image_size = 32
    # elif args.dataset == DatasetType.IMAGENET.value:
    #     base_image_size = 64
    # else:
    #     base_image_size = args.base_image_size
        
    base_image_size = args.base_image_size
    target_image_size = args.target_image_size

    params = {
        TrainingType.ACTIVE_LEARNING: Params(
            batch_size=args.al_batch_size, 
            image_size=target_image_size, 
            lr=args.al_lr, 
            epochs=args.al_epochs,
            optimizer=args.al_optimizer,
            weight_decay=args.al_weight_decay
            ),
        TrainingType.AL_FINETUNING: Params(
            batch_size=args.al_finetune_batch_size, 
            image_size=target_image_size, 
            lr=args.al_lr, 
            epochs=args.al_epochs,
            optimizer=args.target_optimizer,
            weight_decay=args.al_weight_decay
            ),
        TrainingType.BASE_PRETRAIN: Params(
            batch_size=args.base_batch_size, 
            image_size=base_image_size, 
            lr=args.base_lr, 
            epochs=args.base_epochs,
            optimizer=args.base_optimizer,
            weight_decay=args.weight_decay
            ),
        TrainingType.TARGET_PRETRAIN: Params(
            batch_size=args.target_batch_size, 
            image_size=target_image_size, 
            lr=args.target_lr, 
            epochs=args.target_epochs,
            optimizer=args.target_optimizer,
            weight_decay=args.weight_decay
            ),
        TrainingType.FINETUNING: Params(
            batch_size=args.finetune_batch_size, 
            image_size=args.finetune_image_size, 
            lr=args.finetune_lr, 
            epochs=args.finetune_epochs,
            optimizer=args.finetune_optimizer,
            weight_decay=args.finetune_weight_decay
            ),
    }
    return params[training_type]

def accuracy(loss, corrects, loader):
    epoch_loss = loss / len(loader.dataset)
    epoch_acc = corrects.double() / len(loader.dataset)

    return epoch_loss, epoch_acc

def split_dataset(args, dir, transforms, ratio=0.6, is_classifier=False):
    dataset = torchvision.datasets.ImageFolder(
        dir,
        transform=transforms)

    train_ds = dataset
    val_ds = None
    if args.dataset == DatasetType.IMAGENET_LITE.value or is_classifier:
        train_size = int(ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_ds, val_ds = random_split(dataset=dataset, lengths=[train_size, val_size])

    return train_ds, val_ds


def get_ds_num_classes(dataset):
    if dataset == DatasetType.CLIPART.value:
        num_classes = 345
        dir = "/clipart"
        
    elif dataset == DatasetType.SKETCH.value:
        num_classes = 345
        dir = "/sketch"

    elif dataset == DatasetType.QUICKDRAW.value:
        num_classes = 345
        dir = "/quickdraw"

    elif dataset == DatasetType.UCMERCED.value:
        num_classes = 21
        dir = "/ucmerced/images"

    elif dataset == DatasetType.IMAGENET.value:
        num_classes = 200
        dir = "/imagenet"

    elif dataset == DatasetType.IMAGENET_LITE.value:
        num_classes = 100
        dir = "/imagenet"

    else:
        num_classes = 10
        dir = "/cifar10"
    
    return num_classes, dir

def free_mem(X, y):
    del X
    del y
    gc.collect()
    torch.cuda.empty_cache()