import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse

# import cv2
from torch.utils.tensorboard import SummaryWriter
from models.active_learning.pretext_trainer import PretextTrainer
from utils.commons import load_path_loss, load_saved_state, simple_load_model
from utils.random_seeders import set_random_seeds

from utils.yaml_config_hook import yaml_config_hook
from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from models.trainers.sup_pretrainer import SupPretrainer
from models.trainers.classifier import Classifier
import utils.logger as logging
# import logging

logging.init()

def main():
    writer = SummaryWriter()

    if args.ml_project:
        state = load_saved_state(args, pretrain_level="1")
        if not state:
            # pretrainer = SelfSupPretrainer(args, writer)
            pretrainer = SupPretrainer(args, writer)
            pretrainer.first_pretrain()

        # state = simple_load_model(args, path=f'proxy_{args.al_batches-2}.pth')
        # if not state:
        #     pretext = PretextTrainer(args, writer)
        #     pretrain_data = pretext.do_active_learning()
        
        # else:
        #     pretrain_data = load_path_loss(args, args.pretrain_path_loss_file)

        classifier = Classifier(args, writer, pretrain_level="1") #TODO Revert to pretrain_level="AL"
        # classifier.finetune(pretrain_data) 
        classifier.finetune() #Using the AL filtered images to train the classifier won't work for cifar10 since we don't know the class of the images

    else:
        if args.base_pretrain:
            pretrainer = SelfSupPretrainer(args, writer)
            pretrainer.first_pretrain()

        if args.target_pretrain:
            pretrainer = SelfSupPretrainer(args, writer)
            pretrainer.second_pretrain()

        if args.finetune:
            classifier = Classifier(args, writer)
            classifier.finetune()

if __name__ == "__main__":
    #logging.init()

    #raise RuntimeError("Test unhandled")

    parser = argparse.ArgumentParser(description="CASL")
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

    main()

    logging.info("CASL ended.")

