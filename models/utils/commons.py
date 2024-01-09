import glob
import os
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import random_split
import gc
from datautils.dataset_enum import DatasetType, get_dataset_enum

from models.self_sup.simclr.loss.dcl_loss import DCL
from models.self_sup.simclr.loss.nt_xent_loss import NTXentLoss
from models.self_sup.simclr.simclr import SimCLR
from models.self_sup.simclr.simclr_v2 import SimCLRV2
from models.utils.ssl_method_enum import SSL_Method, get_ssl_method
from models.utils.training_type_enum import Params, TrainingType
from utils.commons import get_state_for_da, load_chkpts, load_saved_state
import utils.logger as logging


def get_model_criterion(args, encoder, training_type=TrainingType.ACTIVE_LEARNING, num_classes=4):
    n_features = get_feature_dimensions_backbone(args)

    if training_type == TrainingType.ACTIVE_LEARNING:
        criterion = nn.CrossEntropyLoss()
        model = encoder
        model.linear = nn.Linear(n_features, num_classes)
        print("using Regular model for AL ")

    # this is a tech debt to figure out why AL complains when we do model.fc instead of model.linear

    elif training_type == TrainingType.LINEAR_CLASSIFIER:
        criterion = nn.CrossEntropyLoss()
        model = encoder
        model.fc = nn.Linear(n_features, num_classes)
        print("using Regular model for LC")

    else:
        if args.method == SSL_Method.SIMCLR.value:
            criterion = NTXentLoss(args)
            model = SimCLR(encoder, args.projection_dim, n_features)
            print("using SIMCLR")

        elif args.method == SSL_Method.DCL.value:
            criterion = DCL(args)
            model = SimCLRV2(n_features)
            print("using SIMCLRv2")

    return model, criterion

def get_feature_dimensions_backbone(args):
    if args.backbone == 'resnet18':
        return 512

    elif args.backbone == 'resnet50':
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
    base_image_size = args.base_image_size
    target_image_size = args.target_image_size

    batch_size = args.swav_batch_size
    epochs = args.base_epochs
    temperature = args.swav_temperature
    optimizer = args.swav_optimizer
    base_lr = 0.01
    target_lr = args.swav_base_lr

    params = {
        TrainingType.ACTIVE_LEARNING: Params(
            batch_size=args.al_finetune_batch_size, 
            image_size=target_image_size, 
            lr=args.al_lr, 
            epochs=args.al_epochs,
            optimizer=args.al_optimizer,
            weight_decay=args.al_weight_decay,
            temperature=temperature
            ),
        TrainingType.BASE_PRETRAIN: Params(
            batch_size= batch_size, #16,
            image_size=base_image_size, 
            lr=base_lr, 
            epochs=epochs,
            optimizer=optimizer,
            weight_decay=args.weight_decay,
            temperature=temperature
            ),
        TrainingType.TARGET_PRETRAIN: Params(
            batch_size=batch_size, 
            image_size=target_image_size, 
            lr=target_lr, 
            epochs=args.target_epochs,
            optimizer=optimizer,
            weight_decay=args.weight_decay,
            temperature=temperature
            ),
        TrainingType.LINEAR_CLASSIFIER: Params(
            batch_size=args.lc_batch_size, 
            image_size=args.lc_image_size, 
            lr=args.lc_lr, 
            epochs=args.lc_epochs,
            optimizer=args.lc_optimizer,
            weight_decay=args.weight_decay,
            temperature=temperature
            ),
    }
    params[TrainingType.TARGET_AL] = params[TrainingType.TARGET_PRETRAIN]
    return params[training_type]

def accuracy(loss, corrects, loader):
    epoch_loss = loss / len(loader.dataset)
    epoch_acc = corrects.double() / len(loader.dataset)

    return epoch_loss, epoch_acc

def split_dataset(args, dir, transforms, ratio=0.6, is_classifier=False):
    dataset = torchvision.datasets.ImageFolder(
        dir,
        transform=transforms)

    return split_dataset2(dataset, ratio, is_classifier)

def split_dataset2(dataset, ratio=0.6, is_classifier=False):
    train_ds = dataset
    val_ds = None
    if is_classifier:
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

    elif dataset == DatasetType.AMAZON.value:
        num_classes = 31
        dir = "/office-31/amazon/images"

    elif dataset == DatasetType.WEBCAM.value:
        num_classes = 31
        dir = "/office-31/webcam/images"

    elif dataset == DatasetType.DSLR.value:
        num_classes = 31
        dir = "/office-31/dslr/images"

    elif dataset == DatasetType.PAINTING.value:
        num_classes = 345
        dir = "/painting"

    elif dataset == DatasetType.ARTISTIC.value:
        num_classes = 65
        dir = "/officehome/artistic"

    elif dataset == DatasetType.CLIP_ART.value:
        num_classes = 65
        dir = "/officehome/clip_art"

    elif dataset == DatasetType.PRODUCT.value:
        num_classes = 65
        dir = "/officehome/product"

    elif dataset == DatasetType.REAL_WORLD.value:
        num_classes = 65
        dir = "/officehome/real_world"
    
    return num_classes, dir

def prepare_model(args, trainingType, model):
    params_to_update = model.parameters()
            
    if trainingType == TrainingType.BASE_PRETRAIN:
        logging.info("Using downloaded swav pretrained model")
        model = load_chkpts(args, "swav_800ep_pretrain.pth.tar", model)

    elif trainingType == TrainingType.TARGET_PRETRAIN:
        state = load_saved_state(args, dataset=get_dataset_enum(args.base_dataset), pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)

    elif trainingType == TrainingType.TARGET_AL:
        state = load_saved_state(args, dataset=get_dataset_enum(args.target_dataset), pretrain_level="2")

        # This is the first AL cycle
        if state is None:
            state = load_saved_state(args, dataset=get_dataset_enum(args.base_dataset), pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)

    # freeze some layers
    for name, param in model.named_parameters():
        if 'projection_head' in name or 'prototypes' in name:
            continue

        if 'bn' in name and 'bias' in name or ('layer4' in name and 'bn' in name and 'weight' in name):
            continue

        param.requires_grad = False

    params_to_update = get_params_to_update(model, feature_extract=True)

    return model, params_to_update

def get_images_pathlist(dir, with_train):
    if dir == "./datasets/modern_office_31":
        return glob.glob(dir + '/*/*/*')

    if "./datasets/generated" in dir.split('_'):
        img_path = glob.glob(dir + '/*')

    elif with_train:
        img_path = glob.glob(dir + '/train/*/*')
    else:
        img_path = glob.glob(dir + '/*/*')

    return img_path

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def free_mem(X, y):
    del X
    del y
    gc.collect()
    torch.cuda.empty_cache()