#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import argparse

from datautils.dataset_enum import get_dataset_enum
from models.active_learning.pretext_trainer import PretextTrainer
from utils.random_seeders import set_random_seeds

from utils.yaml_config_hook import yaml_config_hook
from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from models.trainers.classifier import Classifier
import utils.logger as logging

from models.gan.train import do_gen_ai
from models.utils.visualizations.tsne4 import tsne_similarity

logging.init()

def no_pretraining(args, writer):
    args.training_type = "no_prt"
    args.do_gradual_base_pretrain = False
    args.base_pretrain = False
    args.target_pretrain = True

    args.target_epochs = 600

    datasets = [0, 1, 2, 3, 4, 5, 6, 8] 

    for ds in datasets:
        args.target_dataset = ds
        args.lc_dataset = ds

        pretrainer = SelfSupPretrainer(args, writer)
        pretrainer.second_pretrain()

        classifier = Classifier(args, pretrain_level="2" if args.target_pretrain else "1")
        classifier.train_and_eval()

def proxy_source(args, writer):
    # this is for proxy source(B-P-T-F)
    args.do_gradual_base_pretrain = False
    args.base_pretrain = True
    args.target_pretrain = True

    args.training_type = "proxy_source"

    datasets = [0, 1, 2, 3, 4, 5, 8] 

    for ds in datasets:
        args.base_dataset = f'generated_{get_dataset_enum(ds)}'
        args.target_dataset = ds
        args.lc_dataset = ds

        run_proxy_source_sequence(args, writer)

def run_proxy_source_sequence(args, writer):
    if args.target_pretrain:
        pretrainer = SelfSupPretrainer(args, writer)
        pretrainer.second_pretrain()

    classifier = Classifier(args, pretrain_level="2" if args.target_pretrain else "1")
    classifier.train_and_eval()

def single_iter(args, writer):
    # this is for single iteration pretraining with GAN (B-T-F)
    args.do_gradual_base_pretrain = False
    args.base_pretrain = False
    args.target_pretrain = True

    args.training_type = "single_iter"

    datasets = [0, 1, 2, 3, 4, 5, 8] 

    for ds in datasets:
        args.base_dataset = f'generated_{get_dataset_enum(ds)}'
        args.target_dataset = ds
        args.lc_dataset = ds

        run_single_iter_sequence(args, writer)

def run_single_iter_sequence(args, writer):
    if args.base_pretrain:
            pass

    if args.target_pretrain:
        pretrainer = SelfSupPretrainer(args, writer)
        pretrainer.second_pretrain()

    classifier = Classifier(args, pretrain_level="2" if args.target_pretrain else "1")
    classifier.train_and_eval()

def office_dataset(args, writer):
    args.do_gradual_base_pretrain = True
    args.base_pretrain = True
    args.target_pretrain = False

    args.training_type = "office"

    # bases = [9, 9, 11, 11, 10, 10] # A-D, A-W, D-A, D-W, W-A, W-D
    # targs = [11, 10, 9, 10, 9, 11]

    bases = [12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15] # A-C, A-P, A-R, C-A, C-P, C-R, P-A, P-C, P-R, R-A, R-C, R-P
    targs = [13, 14, 15, 12, 14, 15, 12, 13, 15, 12, 13, 14]

    datasets = [9, 10, 11, 12, 13, 14, 15]

    for ds in datasets:
        args.base_dataset = ds
        args.target_dataset = ds

        do_gen_ai(args)

        pretrainer = SelfSupPretrainer(args, writer)
        pretrainer.second_pretrain()

    for i in range(len(bases)):
        run_office_sequence(args, writer, bases[i], targs[i])

def run_office_sequence(args, writer, base, target):
    args.base_dataset = base
    args.target_dataset = target
    args.lc_dataset = target

    pretext = PretextTrainer(args, writer)
    pretext.do_active_learning()

    classifier = Classifier(args, pretrain_level="2" if args.target_pretrain else "1")
    classifier.train_and_eval()

def main(args):
    writer = None

    office_dataset(args, writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GASP")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    args.epoch_num = args.base_epochs
    args.target_epoch_num = args.target_epochs

    set_random_seeds(random_seed=args.seed)

    assert args.target_dataset == args.lc_dataset
    assert args.base_dataset == args.target_dataset

    main(args)
    # tsne_similarity(args)

    logging.info("GASP ended.")

