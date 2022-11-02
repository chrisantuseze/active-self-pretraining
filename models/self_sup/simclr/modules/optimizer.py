'''
Adapted from 

simclr pytorch repo
'''

import torch
from models.utils.training_type_enum import TrainingType

from .lars import LARS


def load_optimizer(args, params, state, train_params):
    scheduler = None
    
    if args.optimizer == "Adam": #TODO Use a lr scheduler to vary the lr
        optimizer = torch.optim.Adam(params, lr=train_params.lr, weight_decay=args.weight_decay)

    elif args.optimizer == "SGD":
        optimizer = torch.optimSGD(params, lr=train_params.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.optimizer == "LARS":
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