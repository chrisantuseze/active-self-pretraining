import os
from sys import prefix
import torch

import pickle

from models.utils.ssl_method_enum import Method


def save_state(args, model, optimizer, pretrain_level="1"):
    if args.method == Method.SIMCLR.value:
        prefix = "simclr"
        optimizer_type = args.optimizer
    else:
        prefix = "myow"
        optimizer_type = args.optimizer

    out = os.path.join(args.model_path, "{}_{}_checkpoint_{}.tar".format(prefix, pretrain_level, args.current_epoch))

    state = {
        'model': model.state_dict(),
        optimizer_type + '-optimizer': optimizer.state_dict()
    }
    torch.save(state, out)

    print("checkpoint saved at {}".format(out))
    args.resume = out

def load_saved_state(args, recent=True, pretrain_level="1"):
    if args.method == Method.SIMCLR.value:
        prefix = "simclr"
    else:
        prefix = "myow"

    model_fp = args.resume if recent and args.resume else os.path.join(
            args.model_path, "{}_{}_checkpoint_{}.tar".format(prefix, pretrain_level, args.epoch_num)
        )

    return torch.load(model_fp, map_location=args.device.type)


def simple_save_model(args, model, path):
    state = {
        'model': model.state_dict()
    }

    out = os.path.join(args.model_path, path)
    torch.save(state, out)

def simple_load_model(args, path):
    out = os.path.join(args.model_path, path)
    checkpoint = torch.load(out)

    return checkpoint['model']

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
    out = os.path.join(args.model_path, filename)

    try:
        with open(out, "wb") as file:
            pickle.dump(image_loss_list, file)

    except IOError:
        print("File could not be opened for write operation")


def load_path_loss(args, filename):
    out = os.path.join(args.model_path, filename)

    try:
        with open(out, "rb") as file:
            return pickle.load(file)

    except IOError:
        return None
