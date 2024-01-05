#!/usr/bin/env python3

import os
from models.trainers.domain_adapter import DomainAdapter
import torch
# from torch.utils.tensorboard import SummaryWriter
import argparse

from utils.random_seeders import set_random_seeds

import utils.logger as logging
from utils.viz import viz
logging.init()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--source_batch_size', default=8, type=int, help='')
    parser.add_argument('--source_image_size', default=64, type=int, help='')
    parser.add_argument('--source_lr', default=1e-3, type=float, help='')
    parser.add_argument('--source_epochs', default=600, type=int, help='')
    parser.add_argument('--source_weight_decay', default=1.0e-4, type=float, help='')
    parser.add_argument('--source_dataset', default=5, type=int, help='')

    parser.add_argument('--target_batch_size', default=4, type=int, help='')
    parser.add_argument('--target_image_size', default=64, type=int, help='')
    parser.add_argument('--target_lr', default=1.0e-4, type=float, help='')
    parser.add_argument('--target_epochs', default=50, type=int, help='')
    parser.add_argument('--target_weight_decay', default=1.0e-4, type=float, help='')
    parser.add_argument('--target_dataset', default=6, type=int, help='')

    parser.add_argument('--al_batch_size', default=256, type=int, help='')
    parser.add_argument('--al_image_size', default=256, type=int, help='')
    parser.add_argument('--al_lr', default=0.01, type=float, help='')
    parser.add_argument('--al_epochs', default=20, type=int, help='')
    parser.add_argument('--al_weight_decay', default=5.0e-4, type=float, help='')
    parser.add_argument('--al_trainer_sample_size', default=400, type=int, help='specifies the amount of samples to be added to the training pool after each AL iteration')
    parser.add_argument('--al_sample_percentage', default=0.95, type=float, help='specifies the percentage of the samples to be used for the target pretraining')
    parser.add_argument('--al_batches', default=10, type=int, help='specifies the number of AL iterations')

    parser.add_argument('--gan_batch_size', default=32, type=int, help='')
    parser.add_argument('--gan_image_size', default=64, type=int, help='')
    parser.add_argument('--gan_lr', default=1.0e-4, type=float, help='')
    parser.add_argument('--gan_epochs', default=100, type=int, help='')
    parser.add_argument('--gan_weight_decay', default=1.0e-6, type=float, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default="./datasets", type=str, help='')
    parser.add_argument('--backbone', default="resnet50", type=str, help='')
    parser.add_argument('--model_checkpoint_path', default="save/checkpoints", type=str, help='')
    parser.add_argument('--model_misc_path', default="save/misc", type=str, help='')
    parser.add_argument('--gen_images_path', default="generated", type=str, help='')

    parser.add_argument('--split_ratio', default=0.8, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--workers', type=int, default=8, help='')
    
    parser.add_argument('--log_step', default=200, type=int, help='')
    parser.add_argument('--seed', default=6, type=int, help='')

    # swav
    parser.add_argument('--crops_for_assign', default=[0, 1], type=int, help='')
    parser.add_argument('--swav_temperature', default=0.1, type=float, help='')
    parser.add_argument('--epsilon', default=0.05, type=float, help='')
    parser.add_argument('--sinkhorn_iterations', default=3, type=int, help='')
    parser.add_argument('--feat_dim', default=128, type=int, help='')
    parser.add_argument('--nmb_prototypes', default=3000, type=int, help='')
    parser.add_argument('--queue_length', default=0, type=int, help='')
    parser.add_argument('--epoch_queue_starts', default=15, type=int, help='')
    parser.add_argument('--hidden_mlp', default=1024, type=int, help='')
    parser.add_argument('--nmb_crops', default=[2], type=int, help='')
    parser.add_argument('--size_crops', default=[224], type=int, help='')
    parser.add_argument('--min_scale_crops', default=[0.14], type=float, help='')
    parser.add_argument('--max_scale_crops', default=[1], type=int, help='')
    parser.add_argument('--world_size', default=-1, type=int, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--local_rank', default=0, type=int, help='')
    parser.add_argument('--final_lr', default=0, type=int, help='')
    parser.add_argument('--freeze_prototypes_niters', default=313, type=int, help='')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='')
    parser.add_argument('--start_warmup', default=0, type=int, help='')

    return parser.parse_args()


def main(args):
    writer = None #SummaryWriter()

    adapter = DomainAdapter(args, writer)
    adapter.train_source()

    adapter.generate_data()

    adapter.train_target()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_checkpoint_path, exist_ok=True)
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")

    set_random_seeds(random_seed=args.seed)

    main(args)

    # viz(args)

    logging.info("GASP ended.")

