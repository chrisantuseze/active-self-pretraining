# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os

import numpy as np
import torch

from models.utils.logger import PD_Stats, create_logger

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.model_misc_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.model_misc_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    # training_stats = PD_Stats(
    #     os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    # )
    training_stats = PD_Stats(
        os.path.join(params.model_misc_path, "stats_" + str(len(args)) + ".pkl"), args
    )

    return training_stats

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res