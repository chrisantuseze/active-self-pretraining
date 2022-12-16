# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from datautils.finetune_dataset import Finetune
from models.self_sup.swav.utils import accuracy, initialize_exp
from models.utils.commons import AverageMeter, get_ds_num_classes, get_params
from models.utils.training_type_enum import TrainingType
from optim.optimizer import load_optimizer
from utils.commons import load_classifier_chkpts
import utils.logger as logging
import models.self_sup.swav.backbone.resnet50 as resnet_models

class Classifier2():
    def __init__(self, args, pretrain_level="2") -> None:
        self.args = args

        self.best_acc = 0.0

        self.training_stats = initialize_exp(
            args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
        )

        # build model
        num_classes, self.dir = get_ds_num_classes(self.args.lc_dataset)
        model = resnet_models.__dict__[args.backbone](output_dim=0, eval_mode=True)
        linear_classifier = LogReg(num_classes, args.backbone, args.global_pooling, args.use_bn)

        # model to gpu
        self.model = model.to(self.args.device)
        self.linear_classifier = linear_classifier.to(self.args.device)

        self.model.eval()

        # load weights
        self.model = load_classifier_chkpts(self.args, self.model, pretrain_level)

        train_params = get_params(self.args, TrainingType.FINETUNING)
        self.optimizer, self.scheduler = load_optimizer(self.args, self.linear_classifier.parameters(), train_params=train_params)
        
        cudnn.benchmark = True

    def train_and_eval(self):
        train_loader, val_loader = Finetune(
            self.args, dir=self.dir, 
            training_type=TrainingType.FINETUNING).get_loader(pretrain_data=None)
            
        for epoch in range(0, self.args.lc_epochs):

            scores = self.train(self.model, self.linear_classifier, self.optimizer, train_loader, epoch)
            scores_val = self.validate_network(val_loader, self.model, self.linear_classifier)
            self.training_stats.update(scores + scores_val)

            self.scheduler.step()

            # save checkpoint
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": self.linear_classifier.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_acc": self.best_acc,
            }
            torch.save(save_dict, os.path.join(self.args.model_checkpoint_path, 'classifier_{:4f}_acc.pth'.format(self.best_acc)))

            logging.info("Training of the supervised linear classifier on frozen features completed.\n"
                    "Top-1 test accuracy: {acc:.1f}\n".format(acc=self.best_acc))

    def train(self, model, reglog, optimizer, loader, epoch):
        """
        Train the models on the dataset.
        """
        # running statistics
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # training statistics
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        end = time.perf_counter()

        model.eval()
        reglog.train()
        criterion = nn.CrossEntropyLoss().cuda()

        for step, (inp, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.perf_counter() - end)

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = model(inp)
            output = reglog(output)

            # compute cross entropy loss
            loss = criterion(output, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            # update stats
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # verbose
            if step % self.args.log_step == 0:
                logging.info(
                    "Epoch[{0}] - Iter: [{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                    "LR {lr}".format(
                        epoch,
                        step,
                        len(loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

        return epoch, losses.avg, top1.avg.item(), top5.avg.item()


    def validate_network(self, val_loader, model, linear_classifier):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        linear_classifier.eval()

        criterion = nn.CrossEntropyLoss().cuda()

        with torch.no_grad():
            end = time.perf_counter()
            for step, (inp, target) in enumerate(val_loader):

                # move to gpu
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                output = linear_classifier(model(inp))
                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), inp.size(0))
                top1.update(acc1[0], inp.size(0))
                top5.update(acc5[0], inp.size(0))

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()

        if top1.avg.item() > self.best_acc:
            self.best_acc = top1.avg.item()

        logging.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=self.best_acc))
            

        return losses.avg, top1.avg.item(), top5.avg.item()


class LogReg(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(LogReg, self).__init__()
        self.bn = None
        if global_avg:
            s = 2048
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
