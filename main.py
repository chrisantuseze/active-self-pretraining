#!/usr/bin/env python3

import os
import torch
import argparse

from models.utils.visualizations.viz import viz
from models.active_learning.pretext_trainer import PretextTrainer
from utils.random_seeders import set_random_seeds

from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from models.trainers.classifier import Classifier
import utils.logger as logging
from datautils import dataset_enum

from models.gan.train import do_gen_ai
from utils.yaml_config_hook import yaml_config_hook

logging.init()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nodes', default=1, type=int, help='')
    parser.add_argument('--gpus', default=1, type=int, help='')

    parser.add_argument('--source_batch_size', default=16, type=int, help='')
    parser.add_argument('--source_image_size', default=256, type=int, help='')
    parser.add_argument('--source_lr', default=1e-3, type=float, help='')
    parser.add_argument('--source_epochs', default=75, type=int, help='') #600
    parser.add_argument('--source_weight_decay', default=1.0e-4, type=float, help='')
    parser.add_argument('--source_dataset', default=7, type=int, help='')

    parser.add_argument('--target_batch_size', default=4, type=int, help='')
    parser.add_argument('--target_image_size', default=256, type=int, help='')
    parser.add_argument('--target_lr', default=1.0e-3, type=float, help='')
    parser.add_argument('--target_epochs', default=75, type=int, help='')
    parser.add_argument('--target_weight_decay', default=1.0e-4, type=float, help='')
    parser.add_argument('--target_dataset', default=8, type=int, help='')

    parser.add_argument('--lc_batch_size', default=32, type=int, help='')
    parser.add_argument('--lc_image_size', default=256, type=int, help='')
    parser.add_argument('--lc_lr', default=0.1, type=float, help='') #1e-3
    parser.add_argument('--lc_epochs', default=200, type=int, help='')
    parser.add_argument('--lc_dataset', default=8, type=int, help='')
    parser.add_argument('--lc_optimizer', default="Classifier", type=str, help='')
    parser.add_argument('--lc_gamma', type=int, default=0.1, help='')
    parser.add_argument('--lc_final_lr', type=int, default=0, help='')

    parser.add_argument('--al_batch_size', default=256, type=int, help='')
    parser.add_argument('--al_image_size', default=256, type=int, help='')
    parser.add_argument('--al_lr', default=0.1, type=float, help='')
    parser.add_argument('--al_epochs', default=20, type=int, help='')
    parser.add_argument('--al_weight_decay', default=5.0e-4, type=float, help='')
    parser.add_argument('--sampling_size', default=400, type=int, help='specifies the amount of samples to be added to the training pool after each AL iteration')
    parser.add_argument('--al_sample_percentage', default=0.95, type=float, help='specifies the percentage of the samples to be used for the target pretraining')
    parser.add_argument('--al_batches', default=8, type=int, help='specifies the number of AL iterations')
    parser.add_argument('--al_finetune_batch_size', default=64, type=int, help='')
    parser.add_argument('--al_optimizer', default="SGD-MultiStepV2", type=str, help='')
    parser.add_argument('--al_path_loss_file', default="al_path_loss.pkl", type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default="./datasets", type=str, help='')
    parser.add_argument('--backbone', default="resnet50", type=str, help='')
    parser.add_argument('--model_checkpoint_path', default="save/checkpoints", type=str, help='')
    parser.add_argument('--model_misc_path', default="save/misc", type=str, help='')

    parser.add_argument('--split_ratio', default=0.8, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--workers', type=int, default=8, help='')
    parser.add_argument('--nesterov', type=int, default=False, help='')
    parser.add_argument('--scheduler_type', type=str, default="cosine", help='')
    parser.add_argument('--decay_epochs', type=int, default=[60, 80], nargs='+', help='')
    
    parser.add_argument('--log_step', default=200, type=int, help='')
    parser.add_argument('--seed', default=1, type=int, help='')

    # swav
    parser.add_argument('--swav_batch_size', default=64, type=int, help='')
    parser.add_argument('--swav_source_lr', default=2.4, type=int, help='')
    parser.add_argument('--swav_optimizer', default="SwAV", type=str, help='')

    parser.add_argument('--crops_for_assign', default=[0, 1], type=int, nargs='+', help='')
    parser.add_argument('--temperature', default=0.1, type=float, help='')
    parser.add_argument('--epsilon', default=0.05, type=float, help='')
    parser.add_argument('--sinkhorn_iterations', default=3, type=int, help='')
    parser.add_argument('--feat_dim', default=128, type=int, help='')
    parser.add_argument('--nmb_prototypes', default=3000, type=int, help='')
    parser.add_argument('--queue_length', default=0, type=int, help='')
    parser.add_argument('--epoch_queue_starts', default=15, type=int, help='')
    parser.add_argument('--hidden_mlp', default=1024, type=int, help='')
    parser.add_argument('--nmb_crops', default=[2], type=int, nargs='+', help='')
    parser.add_argument('--size_crops', default=[224], type=int, nargs='+', help='')
    parser.add_argument('--min_scale_crops', default=[0.14], type=float,  nargs='+',help='')
    parser.add_argument('--max_scale_crops', default=[1], type=int, nargs='+', help='')
    parser.add_argument('--world_size', default=-1, type=int, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--local_rank', default=0, type=int, help='')
    parser.add_argument('--final_lr', default=0, type=int, help='')
    parser.add_argument('--freeze_prototypes_niters', default=313, type=int, help='')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='')
    parser.add_argument('--start_warmup', default=0, type=int, help='')

    return parser.parse_args()

def train(args, writer):
    pretrainer = SelfSupPretrainer(args, writer)
    # pretrainer.first_pretrain()

    pretext = PretextTrainer(args, writer)
    pretext.do_active_learning()

    classifier = Classifier(args, pretrain_level=f"2_{args.al_batches-1}")
    classifier.train_and_eval()

def main(args):
    writer = None
    train(args, writer)

def regular():
    # You can change dataset from here for ease
    args.source_dataset = 4
    args.target_dataset = 5
    args.lc_dataset = args.target_dataset
    assert args.target_dataset == args.lc_dataset

    if dataset_enum.in_domainnet(args.lc_dataset):
        args.lc_batch_size = 256
        args.lc_lr = 0.5
        args.al_batches = 2
        args.lc_epochs = 120

    main(args)

def iterative_training(data_type):
    for item in data_type:
        args.source_dataset = item[0]
        args.target_dataset = item[1]
        args.lc_dataset = args.target_dataset

        if dataset_enum.in_domainnet(args.lc_dataset):
            args.lc_batch_size = 256
            args.lc_lr = 0.5
            args.al_batches = 1
            args.lc_epochs = 120
            args.target_epochs: 150 

        main(args)

if __name__ == "__main__":
    args = parse_args()

    # parser = argparse.ArgumentParser(description="A3")
    # config = yaml_config_hook("./config/config.yaml")
    # for k, v in config.items():
    #     parser.add_argument(f"--{k}", default=v, type=type(v))

    # args = parser.parse_args()
    
    os.makedirs(args.model_checkpoint_path, exist_ok=True)
    os.makedirs(args.model_misc_path, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    args.epoch_num = args.source_epochs

    set_random_seeds(random_seed=args.seed)

    # reduces the image size 
    args.size_crops = [128]

    regular()
    # viz(args)

    logging.info("A3 ended.")