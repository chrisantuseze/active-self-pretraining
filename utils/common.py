import os
import torch

from utils.method_enum import Method


def save_model(args, model, optimizer):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

def load_model(args):
    if args.method == Method.SIMCLR.value:
        prefix = "simclr"
    elif args.method == Method.MOCO.value:
        prefix = "moco"
    else:
        prefix = "swav"

    model_fp = os.path.join(
            args.model_path, "{}_checkpoint_{}.tar".format(prefix, args.epoch_num)
        )

    return torch.load(model_fp, map_location=args.device.type)