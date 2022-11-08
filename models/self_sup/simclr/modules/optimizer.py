'''
Adapted from 

simclr pytorch repo
'''

import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim import SGD, Adam

from models.utils.training_type_enum import TrainingType

from .lars import LARS


def load_optimizer(args, params, state, train_params):
    scheduler = None
    
    if train_params.optimizer == "Adam":
        optimizer = Adam(params, lr=train_params.lr, weight_decay=args.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

    elif train_params.optimizer == "SGD":
        optimizer = SGD(params, lr=train_params.lr, momentum=args.momentum, nesterov=True)
        
        # step_size: at how many multiples of epoch you decay
        # step_size = 1, after every 1 epoch, new_lr = lr*gamma 
        # gamma = decaying factor
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    elif train_params.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            params,
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_params.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    if args.reload:
            optimizer.load_state_dict(state[args.optimizer + '-optimizer'])
    return optimizer, scheduler