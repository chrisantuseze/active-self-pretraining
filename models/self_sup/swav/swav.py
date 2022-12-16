# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from models.self_sup.swav.utils import AverageMeter, bool_flag, initialize_exp
from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from optim.optimizer import load_optimizer
from backbone import resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=5, type=int, #100
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int, #64
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=4, type=int, #10 Had to reduce to 4 due to dataloader error
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


class SwAVTrainer():
    def __init__(self, args, dataloader, pretrain_level="1", 
        training_type=TrainingType.BASE_PRETRAIN, log_step=500) -> None:
        
        self.args = args
        self.train_loader = dataloader

        self.logger, self.training_stats = initialize_exp(args, "epoch", "loss")

        # build model
        self.model = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        )
        self.model = self.model.to(self.args.device)

        self.train_params = get_params(self.args, training_type)
        self.optimizer, self.scheduler = load_optimizer(
            self.args, self.model.parameters(),
            train_params=self.train_params, 
            train_loader=self.train_loader
        )

        # build the queue
        queue = None
        self.queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(self.queue_path):
            queue = torch.load(self.queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        self.args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

        cudnn.benchmark = True

    def train_epoch(self, epoch):
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # optionally starts a queue
        if self.args.queue_length > 0 and epoch >= self.args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(self.args.crops_for_assign),
                self.args.queue_length // self.args.world_size,
                self.args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = self.train(self.train_loader, self.model, self.optimizer, epoch, self.scheduler, queue)
        self.training_stats.update(scores)

        if queue is not None:
            torch.save({"queue": queue}, self.queue_path)


    def train(self, train_loader, model, optimizer, epoch, lr_schedule, queue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        use_the_queue = False

        end = time.time()
        for it, inputs in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # update learning rate
            iteration = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]

            # normalize the prototypes
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)

            # ============ multi-res forward passes ... ============
            embedding, output = model(inputs)
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(self.args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    # get assignments
                    q = self.distributed_sinkhorn(out)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(self.args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / self.args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                loss += subloss / (np.sum(self.args.nmb_crops) - 1)
            loss /= len(self.args.crops_for_assign)

            # ============ backward and optim step ... ============
            optimizer.zero_grad()
            loss.backward()
            # cancel gradients for the prototypes
            if iteration < self.args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()

            # ============ misc ... ============
            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if self.args.rank ==0 and it % 50 == 0:
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        it,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        return (epoch, losses.avg), queue


    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.args.world_size # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.args.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()