import os
from operator import mod
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse

# import cv2
from torch.utils.tensorboard import SummaryWriter
from models.active_learning.pt4al.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone

from utils.yaml_config_hook import yaml_config_hook
from models.trainers.pretrainer import Pretrainer
from models.trainers.classifier import Classifier

def main():
    writer = SummaryWriter()

    if args.base_pretrain:
        pretrainer = Pretrainer(args, writer)
        pretrainer.first_pretrain()

    if args.target_pretrain:
        pretrainer = Pretrainer(args, writer)
        pretrainer.second_pretrain()

    if args.finetune:
        classifier = Classifier(args, writer)
        classifier.finetune()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    main()