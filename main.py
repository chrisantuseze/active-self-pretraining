#!/usr/bin/env python3

import os
from models.trainers.domain_adapter import DomainAdapter
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

from utils.random_seeders import set_random_seeds

import utils.logger as logging
logging.init()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--source_batch_size', default=8, type=int, help='')
    parser.add_argument('--source_image_size', default=128, type=int, help='')
    parser.add_argument('--source_lr', default=5e-4, type=float, help='')
    parser.add_argument('--source_epochs', default=600, type=int, help='')
    parser.add_argument('--source_weight_decay', default=1.0e-3, type=float, help='')
    parser.add_argument('--source_dataset', default=4, type=int, help='')

    parser.add_argument('--target_batch_size', default=4, type=int, help='')
    parser.add_argument('--target_image_size', default=128, type=int, help='')
    parser.add_argument('--target_lr', default=1.0e-3, type=float, help='')
    parser.add_argument('--target_epochs', default=25, type=int, help='')
    parser.add_argument('--target_weight_decay', default=1.0e-6, type=float, help='')
    parser.add_argument('--target_dataset', default=6, type=int, help='')

    parser.add_argument('--al_batch_size', default=256, type=int, help='')
    parser.add_argument('--al_image_size', default=256, type=int, help='')
    parser.add_argument('--al_lr', default=0.01, type=float, help='')
    parser.add_argument('--al_epochs', default=20, type=int, help='')
    parser.add_argument('--al_weight_decay', default=5.0e-4, type=float, help='')
    parser.add_argument('--al_trainer_sample_size', default=400, type=int, help='specifies the amount of samples to be added to the training pool after each AL iteration')
    parser.add_argument('--al_sample_percentage', default=0.95, type=float, help='specifies the percentage of the samples to be used for the target pretraining')
    parser.add_argument('--al_batches', default=10, type=int, help='')

    parser.add_argument('--gan_batch_size', default=32, type=int, help='')
    parser.add_argument('--gan_image_size', default=64, type=int, help='')
    parser.add_argument('--gan_lr', default=1.0e-4, type=float, help='')
    parser.add_argument('--gan_epochs', default=100, type=int, help='')
    parser.add_argument('--gan_weight_decay', default=1.0e-6, type=float, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default="./datasets", type=str, help='')
    parser.add_argument('--backbone', default="resnet18", type=str, help='')
    parser.add_argument('--model_checkpoint_path', default="save/checkpoints", type=str, help='')
    parser.add_argument('--model_misc_path', default="save/misc", type=str, help='')
    parser.add_argument('--gen_images_path', default="generated", type=str, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    parser.add_argument('--split_ratio', default=0.8, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--workers', type=int, default=8, help='')
    
    parser.add_argument('--log_step', default=200, type=int, help='')
    parser.add_argument('--seed', default=6, type=int, help='')

    return parser.parse_args()


def main(args):
    writer = SummaryWriter()

    adapter = DomainAdapter(args, writer)
    # adapter.train_source()

    adapter.generate_data()

    adapter.train_target()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_checkpoint_path, exist_ok=True)
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")

    set_random_seeds(random_seed=args.seed)

    main(args)
    # tsne_similarity(args)

    logging.info("GASP ended.")

