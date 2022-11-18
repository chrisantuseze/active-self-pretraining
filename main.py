import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse

# import cv2
from torch.utils.tensorboard import SummaryWriter
from models.active_learning.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from utils.commons import load_saved_state, simple_load_model
from utils.random_seeders import set_random_seeds

from utils.yaml_config_hook import yaml_config_hook
from models.trainers.pretrainer import Pretrainer
from models.trainers.classifier import Classifier
#import utils.logger as logging
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

logging.basicConfig(filename="datasets/casl.log", level=logging.INFO)
logging.info("CASL started...")

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def main():
    writer = SummaryWriter()

    if args.base_pretrain:
        state = load_saved_state(args, pretrain_level="1")
        if not state:
            pretrainer = Pretrainer(args, writer)
            pretrainer.first_pretrain()

    if args.ml_project:
        state = simple_load_model(args, path=f'proxy_{args.al_batches-2}.pth')
        if not state:
            pretext = PretextTrainer(args, writer)
            pretrain_data = pretext.do_active_learning()

        classifier = Classifier(args, writer, pretrain_level="AL")
        classifier.finetune()

    else:
        if args.target_pretrain:
            pretrainer = Pretrainer(args, writer)
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

    set_random_seeds(random_seed=args.seed)

    main()

    logging.info("CASL ended.")