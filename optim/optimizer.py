import math
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD, Adam

from models.utils.training_type_enum import Params
import utils.logger as logging


def load_optimizer(args, params, train_params: Params=None, train_loader=None):
    scheduler = None
    
    if train_params.optimizer == "Adam-Cosine":
        optimizer = Adam(params, lr=train_params.lr, weight_decay=train_params.weight_decay)
        # this could be implemented to allow for a restart of the learning rate after a certain number of epochs. To do this, simply call
        # line 34 in the check for the number of epochs
        scheduler = CosineAnnealingLR(optimizer, train_params.epochs, eta_min=0, T_max=200)

    elif train_params.optimizer == "SGD-MultiStep":
        optimizer = SGD(params, lr=train_params.lr, momentum=args.momentum, weight_decay=train_params.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

    elif train_params.optimizer == "SGD-MultiStepV2":
        optimizer = torch.optim.SGD(params, lr=train_params.lr, momentum=args.momentum, weight_decay=train_params.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    elif train_params.optimizer == "SwAV":
        # build optimizer
        optimizer = torch.optim.SGD(
            params,
            lr=0.0001,#0.001,#train_params.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        warmup_lr_schedule = np.linspace(args.start_warmup, train_params.lr, len(train_loader) * args.warmup_epochs)
        iters = np.arange(len(train_loader) * (train_params.epochs - args.warmup_epochs))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (train_params.lr - args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(train_loader) * (train_params.epochs - args.warmup_epochs)))) for t in iters])
        scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    elif train_params.optimizer == "Classifier":
        # set optimizer
        optimizer = torch.optim.SGD(
            params,
            lr=train_params.lr,
            nesterov=args.nesterov,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # set scheduler
        if args.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, args.decay_epochs, gamma=args.gamma
            )
        elif args.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, train_params.epochs, eta_min=args.lc_final_lr
            )

    else:
        raise ValueError

    return optimizer, scheduler