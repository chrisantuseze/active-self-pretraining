import math
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD, Adam

from models.utils.training_type_enum import Params
from .lars import LARS
import utils.logger as logging


def load_optimizer(args, params, state=None, train_params: Params=None, train_loader=None):
    scheduler = None
    
    if train_params.optimizer == "SimCLR":
        optimizer = Adam(params, lr=train_params.lr, weight_decay=train_params.weight_decay)

    elif train_params.optimizer == "Adam-Cosine":
        optimizer = Adam(params, lr=train_params.lr, weight_decay=train_params.weight_decay)
        # this could be implemented to allow for a restart of the learning rate after a certain number of epochs. To do this, simply call
        # line 34 in the check for the number of epochs
        scheduler = CosineAnnealingLR(optimizer, train_params.epochs, eta_min=0, T_max=200)

    elif train_params.optimizer == "DCL":
        lr = train_params.lr * train_params.batch_size/256
        optimizer = SGD(params, lr=lr, momentum=args.momentum, nesterov=True)
    
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
            lr=0.001,#train_params.lr,
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

    elif train_params.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        lr = train_params.lr * train_params.batch_size/256
        optimizer = LARS(
            params,
            lr=lr,
            weight_decay=train_params.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_params.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise ValueError

    return optimizer, scheduler

class SwAVScheduler():
    def __init__(self, args, lr, epochs, loader, optimizer) -> None:
        self.args = args
        self.lr = lr
        self.epochs = epochs
        self.loader = loader
        self.optimizer = optimizer
        self.build_schedule()

    def build_schedule(self):
        warmup_lr_schedule = np.linspace(self.args.start_warmup, self.lr, len(self.loader) * self.args.warmup_epochs)
        iters = np.arange(len(self.loader) * (self.epochs - self.args.warmup_epochs))
        cosine_lr_schedule = np.array([self.args.final_lr + 0.5 * (self.lr - self.args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(self.loader) * (self.epochs - self.args.warmup_epochs)))) for t in iters])
        self.scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def step(self, epoch, step):
        iteration = epoch * len(self.loader) + step

        if self.optimizer.param_groups[0]["lr"] < 1.0e-3 or iteration >= len(self.scheduler):
            self.build_schedule()

        try:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.scheduler[iteration]
        except:
            logging.error(f"IndexError: iteration {iteration}, scheduler length {len(self.scheduler)}")