import os
from sys import prefix
import torch

from utils.method_enum import Method


def save_model(args, model, optimizer):
    if args.method == Method.SIMCLR.value:
        prefix = "simclr"
    elif args.method == Method.MOCO.value:
        prefix = "moco"
    else:
        prefix = "swav"

    out = os.path.join(args.model_path, "{}_checkpoint_{}.tar".format(prefix, args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

    print("checkpoint saved at {}".format(out))
    args.resume = out

def load_model(args, recent=True):
    if args.method == Method.SIMCLR.value:
        prefix = "simclr"
    elif args.method == Method.MOCO.value:
        prefix = "moco"
    else:
        prefix = "swav"

    model_fp = args.resume if recent and args.resume else os.path.join(
            args.model_path, "{}_checkpoint_{}.tar".format(prefix, args.epoch_num)
        )

    return torch.load(model_fp, map_location=args.device.type)

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