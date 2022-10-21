import torch
import torch.nn as nn
from datautils.target_pretrain_dataset import get_target_pretrain_ds
from models.active_learning.pt4al.pretext_dataloader import PretextDataLoader
from models.active_learning.pt4al.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.methods.simclr.modules.optimizer import load_optimizer
from models.utils.commons import compute_loss, get_model_criterion
from utils.commons import load_saved_state, save_state
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

            loss = loss.item()
            if step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss}")

            self.writer.add_scalar("Loss/train_epoch", loss, self.args.global_step)
            self.args.global_step += 1

            loss_epoch += loss
        return loss_epoch

            # compute accuracy, display progress and other stuff

    def base_pretrain(self, model, train_loader, criterion, optimizer, scheduler, pretrain_level="1") -> None:
        print("Training in progress, please wait...")

        resume_epoch = self.args.current_epoch if self.args.resume else int(self.args.epoch_num)
        end_epoch = self.args.base_epochs - resume_epoch

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


    def first_pretrain(self) -> None:
        # initialize ResNet
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        print("=> creating model '{}'".format(self.args.resnet))
        model, criterion = get_model_criterion(self.args, encoder)
            
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
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        if self.args.do_al:
            pretext = PretextTrainer(self.args, encoder)
            pretrain_data = pretext.do_active_learning()
            loader = PretextDataLoader(pretrain_data).get_loader()
        else:
            loader = get_target_pretrain_ds(self.args, isAL=False).get_loader()
        
        model = encoder
        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'])
        model = model.to(self.args.device)

        _, criterion = get_model_criterion(self.args, encoder)
        optimizer, scheduler = load_optimizer(self.args, model, state)
        self.base_pretrain(model, loader, criterion, optimizer, scheduler, pretrain_level="2")