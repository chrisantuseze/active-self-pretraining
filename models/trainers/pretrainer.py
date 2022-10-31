import torch
import torch.nn as nn
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.active_learning.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.methods.simclr.modules.optimizer import load_optimizer
from models.utils.commons import compute_loss, get_model_criterion
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss, load_saved_state, save_state
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
            optimizer.zero_grad()
            loss = compute_loss(self.args, images, model, criterion)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            if step % 100 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss}")

            self.writer.add_scalar("Loss/train_epoch", loss, self.args.global_step)
            self.args.global_step += 1

            loss_epoch += loss
        return loss_epoch

    def base_pretrain(self, model, train_loader, criterion, optimizer, scheduler, lr, epochs, pretrain_level="1") -> None:
        print("Training in progress, please wait...")

        resume_epoch = self.args.current_epoch if self.args.resume else int(self.args.epoch_num)
        end_epoch = epochs - resume_epoch

        best_epoch_loss = 0
        for epoch in range(self.args.start_epoch, end_epoch):
            print('\nEpoch {}/{}'.format(epoch, (end_epoch - self.args.start_epoch)))
            print('-' * 10)
            
            loss_epoch = self.train_single_epoch(model, train_loader, criterion, optimizer)

            # I honestly don't know what this does
            if scheduler is not None:
                scheduler.step()

            if loss_epoch < best_epoch_loss:
                best_epoch_loss = loss_epoch
                save_state(self.args, model, optimizer, pretrain_level)

            self.writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            self.writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch + resume_epoch}/{end_epoch}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            self.args.current_epoch += 1

        save_state(self.args, model, optimizer, pretrain_level)


    def first_pretrain(self) -> None:
        # initialize ResNet
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        print("=> creating model '{}'".format(self.args.resnet))
        model, criterion = get_model_criterion(self.args, encoder, training_type=TrainingType.BASE_PRETRAIN)
            
        state = None
        if self.args.reload:
            state = load_saved_state(self.args, pretrain_level="1")
            model.load_state_dict(state['model'], strict=False)
            
        model = model.to(self.args.device)
        optimizer, scheduler = load_optimizer(self.args, model, state, self.args.base_lr, self.args.base_epochs)

        if self.args.dataset == dataset_enum.DatasetType.IMAGENET.value or self.args.dataset == dataset_enum.DatasetType.IMAGENET_LITE.value:
            train_loader = imagenet.ImageNet(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        elif self.args.dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        else:
            NotImplementedError

        self.base_pretrain(model, train_loader, criterion, optimizer, scheduler, self.args.base_lr, self.args.base_epochs, pretrain_level="1")


    def second_pretrain(self) -> None:
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        if self.args.do_al:

            pretrain_data = None# load_path_loss(self.args, self.args.pretrain_path_loss_file)
            if pretrain_data is None:
                pretext = PretextTrainer(self.args, encoder)
                pretrain_data = pretext.do_active_learning()

            loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN).get_loader()
        else:
            loader = get_target_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loader()        

        model, criterion = get_model_criterion(self.args, encoder, training_type=TrainingType.TARGET_PRETRAIN)

        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        optimizer, scheduler = load_optimizer(self.args, model, state, self.args.target_lr, self.args.target_epochs)
        self.base_pretrain(model, loader, criterion, optimizer, scheduler, self.args.target_lr, self.args.target_epochs, pretrain_level="2")