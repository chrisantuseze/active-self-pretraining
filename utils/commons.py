import os
from sys import prefix
import torch

import pickle
from PIL import Image
from models.active_learning.al_method_enum import get_al_method_enum

from models.utils.ssl_method_enum import SSL_Method
from datautils.dataset_enum import get_dataset_enum
import logging


def save_state(args, model, optimizer, pretrain_level="1", optimizer_type="Adam-Cosine"):
    if args.method == SSL_Method.SIMCLR.value:
        prefix = "simclr"

    elif args.method == SSL_Method.DCL.value:
        prefix = "dcl"

    else:
        prefix = "myow"

    out = os.path.join(args.model_path, "{}_{}_checkpoint_{}.tar".format(prefix, pretrain_level, args.current_epoch))

    state = {
        'model': model.state_dict(),
        optimizer_type + '-optimizer': optimizer.state_dict()
    }
    torch.save(state, out)

    print("checkpoint saved at {}".format(out))
    args.resume = out

def load_saved_state(args, recent=True, pretrain_level="1"):
    if args.method == SSL_Method.SIMCLR.value:
        prefix = "simclr"

    elif args.method == SSL_Method.DCL.value:
        prefix = "dcl"

    else:
        prefix = "myow"

    if pretrain_level == "2":
        epoch_num = args.target_base_epochs

    else:
        epoch_num = args.base_epochs

    out = args.resume if recent and args.resume else os.path.join(
            args.model_path, "{}_{}_checkpoint_{}.tar".format(prefix, pretrain_level, epoch_num)
        )

    return torch.load(out, map_location=args.device.type)


def simple_save_model(args, model, path):
    state = {
        'model': model.state_dict()
    }

    out = os.path.join(args.model_path, path)
    torch.save(state, out)

def simple_load_model(args, path):
    try:
        out = os.path.join(args.model_path, path)
        return torch.load(out)

    except IOError:
        return None

def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def save_path_loss(args, filename, image_loss_list):
    filename = "{}_{}".format(get_dataset_enum(args.target_dataset), filename)
    out = os.path.join(args.model_path, filename)

    try:
        with open(out, "wb") as file:
            pickle.dump(image_loss_list, file)

        logging.info("path loss saved at {out}")

    except IOError:
        print("File could not be opened for write operation")


def load_path_loss(args, filename):
    filename = "{}_{}".format(get_dataset_enum(args.target_dataset), filename)
    out = os.path.join(args.model_path, filename)

    try:
        with open(out, "rb") as file:
            return pickle.load(file)

    except IOError:
        return None

def save_accuracy_to_file(args, accuracies, best_accuracy):
    dataset = f"{get_dataset_enum(args.dataset)}-{get_dataset_enum(args.target_dataset)}-{get_dataset_enum(args.finetune_dataset)}"
    filename = "{}_{}_batch_{}".format(dataset, get_al_method_enum(args.al_method), args.finetune_epochs)
    out = os.path.join(args.model_path, filename)

    try:
        with open(out, "a") as file:
            file.write("The accuracies are: \n")
            file.write(", ".join(accuracies))

            file.write("\nThe best accuracy is: \n")
            file.write(str(best_accuracy))

            logging.info("accuracies saved saved at {out}")

    except IOError:
        print("File could not be opened for write operation")

def load_accuracy_file(args):
    dataset = f"{get_dataset_enum(args.dataset)}-{get_dataset_enum(args.target_dataset)}-{get_dataset_enum(args.finetune_dataset)}"
    filename = "{}_{}_batch_{}".format(dataset, get_al_method_enum(args.al_method), args.finetune_epochs)
    out = os.path.join(args.model_path, filename)

    try:
        with open(out, "a") as file:
            return file.readlines()

    except IOError:
        return None

def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')