#!/usr/bin/env python3

import torch
import argparse

from models.active_learning.pretext_trainer import PretextTrainer
from utils.random_seeders import set_random_seeds

from utils.yaml_config_hook import yaml_config_hook
from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from models.trainers.classifier import Classifier
import utils.logger as logging

from models.gan.train import do_gen_ai

# logging.init()

def office_dataset(args, writer):
    args.do_gradual_base_pretrain = True
    args.base_pretrain = True
    args.target_pretrain = True

    args.base_dataset = 5 #7
    args.target_dataset = 6 #8
    args.lc_dataset = 6 #8

    pretrainer = SelfSupPretrainer(args, writer)
    # pretrainer.first_pretrain()

    # do_gen_ai(args)
    # pretrainer.second_pretrain()

    pretext = PretextTrainer(args, writer)
    # pretext.do_active_learning()

    classifier = Classifier(args, pretrain_level="2")
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

