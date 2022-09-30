import os
from operator import mod
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse
from torch.utils.tensorboard import SummaryWriter
from models.backbones.resnet import resnet_backbone

from utils.common import load_model

from utils.yaml_config_hook import yaml_config_hook
from models.pretrainer import Pretrainer
from models.classifier import Classifier

def main():
    writer = SummaryWriter()

    encoder = resnet_backbone(args.resnet, pretrained=False)

    if args.pretrain:
        pretrainer = Pretrainer(args, writer)
        pretrainer.pretrain(encoder)

    if args.finetune:
        classifier = Classifier(args, writer)
        classifier.train(encoder)

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