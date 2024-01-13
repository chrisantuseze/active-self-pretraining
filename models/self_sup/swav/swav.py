from datautils.dataset_enum import get_dataset_info
from models.active_learning.domain_discriminator import DomainClassifier
from models.utils.asl_sfda import VirtualAdversarialLoss, entropy_loss, weight_reg_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import os
import time

import numpy as np
from models.self_sup.swav.utils import initialize_exp
from models.utils.commons import get_params, AverageMeter, prepare_model
from models.utils.training_type_enum import TrainingType
from optim.optimizer import load_optimizer
from utils.commons import load_saved_state
import utils.logger as logging
import models.self_sup.swav.backbone.resnet50 as resnet_models

class SwAVTrainer():
    def __init__(self, args, dataloader, pretrain_level, training_type=TrainingType.BASE_PRETRAIN, log_step=500) -> None:
        self.args = args
        self.train_loader = dataloader
        self.log_step = log_step
        self.training_type = training_type

        self.training_stats = initialize_exp(args, "epoch", "loss")

        # build model
        zero_init_residual = True #This improves the network by 0.2-0.3%
        encoder = resnet_models.__dict__[args.backbone](
            zero_init_residual=zero_init_residual,
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        )

        # load weights

        self.model, params_to_update = prepare_model(self.args, training_type, pretrain_level, encoder) 
        self.model = self.model.to(self.args.device)

        self.train_params = get_params(self.args, training_type)
        self.optimizer, self.scheduler = load_optimizer(
            self.args, params_to_update,
            train_params=self.train_params, 
            train_loader=self.train_loader
        )

        # build the queue
        self.queue = None
        self.queue_path = os.path.join(args.model_misc_path, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        self.args.queue_length -= args.queue_length % (args.swav_batch_size * args.world_size)

        cudnn.benchmark = True

        if training_type == TrainingType.TARGET_AL:
            # self.virt_adv_loss = VirtualAdversarialLoss()

            self.source_model = encoder
            state = load_saved_state(args, dataset=get_dataset_info(args.base_dataset)[1], pretrain_level="1")
            self.source_model.load_state_dict(state['model'], strict=False)
            self.source_model = self.source_model.to(args.device)
            self.source_model.eval()

            self.domain_classifier = DomainClassifier(in_feature=128).to(args.device)

    def train_epoch(self, epoch):

        # optionally starts a queue
        if self.args.queue_length > 0 and epoch >= self.args.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                len(self.args.crops_for_assign),
                self.args.queue_length // self.args.world_size,
                self.args.feat_dim,
            ).cuda()

        # train the network
        scores, self.queue = self.train(self.train_loader, epoch, self.queue)
        self.training_stats.update(scores)

        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)


    def train(self, train_loader, epoch, queue):
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
            embedding_, output = self.model(inputs)
            embedding = embedding_.detach()
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

            #########################################################
            if self.training_type == TrainingType.TARGET_AL:
                s_embedding, _ = self.source_model(inputs)
                src_domain_out = self.domain_classifier(s_embedding)

                tgt_domain_out = self.domain_classifier(embedding_)

                # domain adversarial loss
                # s_loss = F.binary_cross_entropy(src_domain_out, torch.zeros_like(src_domain_out))
                # t_loss = F.binary_cross_entropy(tgt_domain_out, torch.ones_like(tgt_domain_out))
                # print("s_loss", s_loss.item(), "t_loss", t_loss.item())
                # domain_adv_loss = (s_loss + t_loss)/2
                domain_adv_loss = F.binary_cross_entropy(src_domain_out, torch.zeros_like(src_domain_out)) + F.binary_cross_entropy(tgt_domain_out, torch.ones_like(tgt_domain_out))
                domain_adv_loss *= 0.6 * 0.5

                # domain confusion loss
                conf_loss = F.binary_cross_entropy(src_domain_out, torch.ones_like(tgt_domain_out)) + F.binary_cross_entropy(tgt_domain_out, torch.zeros_like(src_domain_out)) 
                # print("conf_loss", conf_loss.item())
                conf_loss *= 0.15 * 0.5

                # Option 2: Entropy maximization  
                s_loss = -torch.sum(F.log_softmax(src_domain_out, dim=0))
                t_loss = -torch.sum(F.log_softmax(tgt_domain_out, dim=0))
                print("s_loss", s_loss.item(), "t_loss", t_loss.item())
                entropy_conf_loss = 0.5 * (s_loss + t_loss)  #-torch.sum(F.log_softmax(src_domain_out, dim=1)) - torch.sum(F.log_softmax(tgt_domain_out, dim=1))
                print("entropy_conf_loss", entropy_conf_loss.item())

                domain_conf_loss = 0.5 * (conf_loss + entropy_conf_loss)


                loss += domain_adv_loss + domain_conf_loss

            #########################################################

            self.optimizer.zero_grad()
            loss.backward()
            # cancel gradients for the prototypes
            if iteration < self.args.freeze_prototypes_niters:
                for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            self.optimizer.step()

            if self.training_type == TrainingType.TARGET_AL:
                # Adjust lambda
                self.domain_classifier.coeff += 0.001

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