import torch
import torch.nn as nn
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.active_learning.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.self_sup.myow.model.resnets import resnet_cifar
from models.self_sup.myow.trainer.myow_trainer import MYOWTrainer, get_myow_trainer
from models.self_sup.simclr.trainer.simclr_trainer import SimCLRTrainer
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss, load_saved_state, save_state
from datautils import dataset_enum, cifar10, imagenet
from torch.utils.tensorboard import SummaryWriter

from models.utils.ssl_method_enum import Method


class Pretrainer:
    def __init__(self, 
        args, 
        writer) -> None:

        self.args = args
        self.writer = writer

    def base_pretrain(self, encoder, train_loader, lr, epochs, trainingType) -> None:
        print("Training in progress, please wait...")

        resume_epoch = self.args.current_epoch if self.args.resume else int(self.args.epoch_num)
        end_epoch = epochs - resume_epoch

        pretrain_level = "1" if trainingType == TrainingType.BASE_PRETRAIN else "2"
        best_epoch_loss = 0
        if self.args.method == Method.SIMCLR.value:
            trainer = SimCLRTrainer(self.args, self.writer, encoder, train_loader, pretrain_level, trainingType)

        elif self.args.method == Method.MYOW.value:
            trainer = get_myow_trainer(self.args, self.writer, encoder, train_loader, pretrain_level, trainingType)

        else:
            NotImplementedError

        model = trainer.model
        optimizer = trainer.optimizer

        for epoch in range(self.args.start_epoch, end_epoch):
            print('\nEpoch {}/{}'.format(epoch, (end_epoch - self.args.start_epoch)))
            print('-' * 10)

            epoch_loss = trainer.train_epoch()

            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_state(self.args, model, optimizer, pretrain_level)

            print(f"Epoch Loss: {epoch_loss / len(train_loader)}\t lr: {round(lr, 5)}")
            print('-' * 10)
            self.args.current_epoch += 1

        save_state(self.args, model, optimizer, pretrain_level)


    def first_pretrain(self) -> None:
        # initialize ResNet
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        print("=> creating model '{}'".format(self.args.resnet))

        if self.args.dataset == dataset_enum.DatasetType.IMAGENET.value or self.args.dataset == dataset_enum.DatasetType.IMAGENET_LITE.value:
            train_loader = imagenet.ImageNet(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        elif self.args.dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        else:
            NotImplementedError

        self.base_pretrain(encoder, train_loader, self.args.base_lr, self.args.base_epochs, trainingType=TrainingType.BASE_PRETRAIN)


    def second_pretrain(self) -> None:
        if self.args.do_al:

            pretrain_data = None# load_path_loss(self.args, self.args.pretrain_path_loss_file)
            if pretrain_data is None:
                pretext = PretextTrainer(self.args)
                pretrain_data = pretext.do_active_learning()

            loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN).get_loader()
        else:
            loader = get_target_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loader()        

        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        self.base_pretrain(encoder, loader, self.args.target_lr, self.args.target_epochs, trainingType=TrainingType.TARGET_PRETRAIN)