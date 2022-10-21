'''
Adapted from 

simclr pytorch repo
'''

import torch

from utils.method_enum import Method
from .lars import LARS


def load_optimizer(args, model, state):

    scheduler = None
    if args.method == Method.MOCO.value:
        # define loss function (criterion) and optimizer
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        
        if args.reload:
            optimizer.load_state_dict(state['moco-optimizer'])
        return optimizer, scheduler

    if args.optimizer == "Adam": #TODO Use a lr scheduler to vary the lr
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    if args.reload:
            optimizer.load_state_dict(state[args.optimizer + '-optimizer'])
    return optimizer, scheduler