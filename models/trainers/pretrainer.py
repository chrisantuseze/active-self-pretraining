import torch
import torch.nn as nn
from models.active_learning.pt4al.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.heads.nt_xent import NT_Xent
from models.methods.moco.moco import MoCo
from models.methods.simclr.simclr import SimCLR
from models.methods.simclr.modules.optimizer import load_optimizer
from models.utils.compute_loss import compute_loss
from utils.common import load_saved_state, save_state
from utils.method_enum import Method
from datautils import dataset_enum, cifar10, imagenet
from torch.utils.tensorboard import SummaryWriter


class Pretrainer:
    def __init__(self, 
        args, 
        writer) -> None:

        self.args = args
        self.writer = writer

    def train_single_epoch(self, model, train_loader, criterion, optimizer) -> int:
        loss_epoch = 0
        model.train()

        for step, (images, _) in enumerate(train_loader):
            loss = compute_loss(self.args, images, model, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            self.writer.add_scalar("Loss/train_epoch", loss.item(), self.args.global_step)
            self.args.global_step += 1

            loss_epoch += loss.item()
        return loss_epoch

            # compute accuracy, display progress and other stuff

    def base_pretrain(self, model, train_loader, criterion, optimizer, scheduler, pretrain_level="1") -> None:
        print("Training in progress, please wait...")

        resume_epoch = self.args.current_epoch if self.args.resume else int(self.args.epoch_num)
        end_epoch = self.args.epochs - resume_epoch

        for epoch in range(self.args.start_epoch, end_epoch):
            loss_epoch = self.train_single_epoch(model, train_loader, criterion, optimizer)

            # I honestly don't know what this does
            if scheduler is not None:
                scheduler.step()

            if epoch % 10 == 0:
                save_state(self.args, model, optimizer, pretrain_level)

            self.writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            self.writer.add_scalar("Misc/learning_rate", self.args.lr, epoch)
            print(
                f"Epoch [{epoch + resume_epoch}/{end_epoch}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(self.args.lr, 5)}"
            )
            self.args.current_epoch += 1

        save_state(self.args, model, optimizer, pretrain_level)


    def first_pretrain(self, encoder) -> None:

        # initialize ResNet
        print("=> creating model '{}'".format(self.args.resnet))

        n_features = encoder.fc.in_features  # get dimensions of fc layer

        if self.args.method == Method.SIMCLR.value:
            criterion = NT_Xent(self.args.batch_size, self.args.temperature, self.args.world_size)
            model =  SimCLR(encoder, self.args.projection_dim, n_features)
            print("using SIMCLR")
            
        elif self.args.method == Method.MOCO.value:
            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().to(self.args.device)
            model = MoCo(encoder, self.args.moco_dim, self.args.moco_k, self.args.moco_m, self.args.moco_t, self.args.mlp)
            print("using MOCO")

        elif self.args.method == Method.SWAV.value:
            NotImplementedError

        else:
            NotImplementedError
            
        state = None
        if self.args.reload:
            state = load_saved_state(self.args, pretrain_level="1")
            model.load_state_dict(state['model'])
            
        model = model.to(self.args.device)
        optimizer, scheduler = load_optimizer(self.args, model, state)

        if self.args.dataset == dataset_enum.DatasetType.IMAGENET.value:
            train_loader = imagenet.ImageNet(self.args).get_loader()

        elif self.args.dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args).get_loader()

        else:
            NotImplementedError

        self.base_pretrain(model, train_loader, criterion, optimizer, scheduler, pretrain_level="1")


    def second_pretrain(self) -> None:
        if self.args.do_al:
            pretext = PretextTrainer(self.args)
            pretext.do_active_learning()

        # continue with second pretraining