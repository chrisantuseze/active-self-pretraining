import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from torch.optim import SGD, Adam

from models.utils.training_type_enum import Params, TrainingType
from .lars import LARS


def load_optimizer(args, params, state, train_params: Params):
    scheduler = None
    
    if train_params.optimizer == "Adam":
        optimizer = Adam(params, lr=train_params.lr, weight_decay=train_params.weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    elif train_params.optimizer == "Adam-Cosine":
        optimizer = Adam(params, lr=train_params.lr, weight_decay=train_params.weight_decay)
        # this could be implemented to allow for a restart of the learning rate after a certain number of epochs. To do this, simply call
        # line 34 in the check for the number of epochs
        scheduler = CosineAnnealingLR(optimizer, eta_min=0.001, T_max=200)

    elif train_params.optimizer == "SGD":
        lr = train_params.lr * train_params.batch_size/256
        optimizer = SGD(params, lr=lr, momentum=args.momentum, nesterov=True)
    
        scheduler = CosineAnnealingLR(optimizer, eta_min=0.001, T_max=200)

    elif train_params.optimizer == "SGD-MultiStep":
        optimizer = SGD(params, lr=train_params.lr, momentum=args.momentum, weight_decay=train_params.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

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

    if args.reload and state:
            optimizer.load_state_dict(state[args.optimizer + '-optimizer'])
    return optimizer, scheduler