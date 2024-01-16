#!/usr/bin/env python3

import torch
import argparse

from models.utils.visualizations.viz import viz
from models.active_learning.pretext_trainer import PretextTrainer
from utils.random_seeders import set_random_seeds

from utils.yaml_config_hook import yaml_config_hook
from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from models.trainers.classifier import Classifier
import utils.logger as logging
from datautils.dataset_enum import DatasetType

from models.gan.train import do_gen_ai

logging.init()

def office_dataset(args, writer):
    args.do_gradual_base_pretrain = True
    args.base_pretrain = True
    args.target_pretrain = True

    pretrainer = SelfSupPretrainer(args, writer)
    pretrainer.first_pretrain()

    # do_gen_ai(args)

    pretext = PretextTrainer(args, writer)
    pretext.do_active_learning()

    # classifier = Classifier(args, pretrain_level=f"2_{args.al_batches-1}")
    classifier = Classifier(args, pretrain_level=f"2_2")
    classifier.train_and_eval()

def main(args):
    writer = None
    office_dataset(args, writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A3")
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

    if args.lc_dataset in [DatasetType.CLIPART.value, DatasetType.SKETCH.value, DatasetType.REAL.value, DatasetType.PAINTING.value]:
        args.lc_batch_size = 128
        args.lc_lr = 0.5
        args.al_batches = 3

    main(args)
    # viz(args)

    logging.info("A3 ended.")

