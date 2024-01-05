import math
from models.trainers.resnet import resnet_backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import os
import time

import numpy as np
from models.self_sup.utils import initialize_exp
from models.utils.commons import get_params, AverageMeter, prepare_model
from models.utils.training_type_enum import TrainingType
from utils.commons import load_chkpts, simple_save_model
import utils.logger as logging
import models.self_sup.resnet50 as resnet_models

class SwAVTrainer():
    def __init__(self, args, model, dataloader, train_params, training_type=TrainingType.SOURCE_PRETRAIN, log_step=500) -> None:
        self.args = args
        self.train_loader = dataloader
        self.log_step = log_step
        self.model = model
        self.train_params = train_params

        self.training_stats = initialize_exp(args, "epoch", "loss")

        # load weights
        if training_type == TrainingType.SOURCE_PRETRAIN:
            logging.info("Using downloaded swav pretrained model")
            self.model = load_chkpts(args, "swav_800ep_pretrain.pth.tar", self.model)
            # self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            print(self.model)

        params_to_update = prepare_model(self.model) 

        self.model = self.model.to(self.args.device)

        # build optimizer
        self.optimizer = torch.optim.SGD(
            params_to_update,
            lr=0.0001,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        warmup_lr_schedule = np.linspace(args.start_warmup, self.train_params.lr, len(self.train_loader) * args.warmup_epochs)
        iters = np.arange(len(self.train_loader) * (self.train_params.epochs - args.warmup_epochs))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (self.train_params.lr - args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(self.train_loader) * (self.train_params.epochs - args.warmup_epochs)))) for t in iters])
        self.scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        # build the queue
        self.queue = None
        self.queue_path = os.path.join(args.model_misc_path, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        self.args.queue_length -= args.queue_length % (self.train_params.batch_size * args.world_size)

        cudnn.benchmark = True

    def train(self) -> None:
        for epoch in range(self.train_params.epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.train_params.epochs))
            logging.info('-' * 20)

            self.train_epoch(epoch)

        simple_save_model(self.args, self.model, f'{self.train_params.name}.pth')

    def train_epoch(self, epoch):

        # optionally starts a queue
        if self.args.queue_length > 0 and epoch >= self.args.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                len(self.args.crops_for_assign),
                self.args.queue_length // self.args.world_size,
                self.args.feat_dim,
            ).cuda()

        # train the network
        scores, self.queue = self.train_step(self.train_loader, epoch, self.queue)
        self.training_stats.update(scores)

        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)


    def train_step(self, train_loader, epoch, queue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()
        use_the_queue = False

        end = time.time()
        for it, inputs in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # update learning rate
            iteration = epoch * len(train_loader) + it
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.scheduler[iteration]

            # normalize the prototypes
            with torch.no_grad():
                w = self.model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.prototypes.weight.copy_(w)

            # ============ multi-res forward passes ... ============
            embedding, output = self.model(inputs)
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
                                self.model.prototypes.weight.t()
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
            self.optimizer.zero_grad()
            loss.backward()
            # cancel gradients for the prototypes
            if iteration < self.args.freeze_prototypes_niters:
                for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            self.optimizer.step()

            # ============ misc ... ============
            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if self.args.rank ==0 and it % self.log_step == 0:
                logging.info(
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
                        lr=self.optimizer.param_groups[0]["lr"],
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