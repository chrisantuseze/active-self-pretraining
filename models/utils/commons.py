import glob
import os
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import random_split
from datautils.dataset_enum import DatasetType

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

    params = {
        TrainingType.ACTIVE_LEARNING: Params(
            batch_size=args.al_batch_size, #doesn't need one though
            image_size=args.al_image_size, 
            lr=args.al_lr, 
            epochs=args.al_epochs,
            weight_decay=args.al_weight_decay,
            name="active_learning",
            ),
        TrainingType.SOURCE_PRETRAIN: Params(
            batch_size=args.source_batch_size,
            image_size=args.source_image_size, 
            lr=args.source_lr, 
            epochs=args.source_epochs,
            weight_decay=args.source_weight_decay,
            name="source",
            ),
        TrainingType.TARGET_PRETRAIN: Params(
            batch_size=args.target_batch_size, 
            image_size=args.target_image_size, 
            lr=args.target_lr, 
            epochs=args.target_epochs,
            weight_decay=args.target_weight_decay,
            name="target",
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
        dir = "/amazon/images"

    elif dataset == DatasetType.WEBCAM.value:
        num_classes = 31
        dir = "/webcam/images"

    elif dataset == DatasetType.DSLR.value:
        num_classes = 31
        dir = "/dslr/images"

    elif dataset == DatasetType.PAINTING.value:
        num_classes = 345
        dir = "/painting"

    elif dataset == DatasetType.ARTISTIC.value:
        num_classes = 65
        dir = "/artistic"

    elif dataset == DatasetType.CLIP_ART.value:
        num_classes = 65
        dir = "/clip_art"

    elif dataset == DatasetType.PRODUCT.value:
        num_classes = 65
        dir = "/product"

    elif dataset == DatasetType.REAL_WORLD.value:
        num_classes = 65
        dir = "/real_world"
    
    return num_classes, dir

def prepare_model(args, trainingType, model):
    params_to_update = model.parameters()
            
    if (trainingType == TrainingType.SOURCE_PRETRAIN and args.base_pretrain) or (trainingType == TrainingType.TARGET_PRETRAIN and not args.base_pretrain):
        state = load_saved_state(args, pretrain_level="1")
        if args.do_gradual_base_pretrain and state is not None:
            logging.info("Using base pretrained model")

            model.load_state_dict(state['model'], strict=False)

        elif args.training_type in ["uc2", "pete_2"]:
            state = get_state_for_da(args)
            model.load_state_dict(state['model'], strict=False)

        else:
            logging.info("Using downloaded swav pretrained model")
            model = load_chkpts(args, "swav_800ep_pretrain.pth.tar", model)

    else:
        state = load_saved_state(args, pretrain_level="1")
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