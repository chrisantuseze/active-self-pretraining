from operator import mod
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse
from models.moco.moco import MoCo
from models.simclr.simclr import SimCLR
from models.simclr import modules

from utils.yaml_config_hook import yaml_config_hook
from models.train import train
from utils.method_enum import Method
from datautils import imagenet, cifar10, dataset_enum

def main():

    # initialize ResNet
    print("=> creating model '{}'".format(args.resnet))

    encoder = modules.get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    if args.method == Method.SIMCLR:
        criterion = modules.NT_Xent(args.batch_size, args.temperature, args.world_size)
        optimizer, scheduler = modules.load_optimizer(args, model)
        model = SimCLR(encoder, args.projection_dim, n_features)
        
    elif args.method == Method.MOCO:
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
                        
        scheduler = None

        model = MoCo(encoder,
                args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    print(model)

    if args.dataset == dataset_enum.DatasetType.IMAGENET:
        train_loader = imagenet.ImageNet()

    elif args.dataset == dataset_enum.DatasetType.CIFAR10:
        train_loader = cifar10.CIFAR10(args)

    elif args.dataset == dataset_enum.DatasetType.STL:
        None

    else:
        NotImplementedError

    model = model.to(args.device)
    
    train(args, model, train_loader, criterion, optimizer, args.method, scheduler)

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