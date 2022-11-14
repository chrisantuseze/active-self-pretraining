import torch
import torch.nn as nn
import utils.logger as logging
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.active_learning.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.self_sup.myow.model.resnets import resnet_cifar
from models.self_sup.myow.trainer.myow_trainer import MYOWTrainer, get_myow_trainer
from models.self_sup.simclr.trainer.simclr_trainer import SimCLRTrainer
from models.self_sup.simclr.trainer.simclr_trainer_v2 import SimCLRTrainerV2
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss, load_saved_state, save_state
from datautils import dataset_enum, cifar10, imagenet
from torch.utils.tensorboard import SummaryWriter

from models.utils.ssl_method_enum import SSL_Method


class Pretrainer:
    def __init__(self, 
        args, 
        writer) -> None:

        self.args = args
        self.writer = writer

    def base_pretrain(self, encoder, train_loader, epochs, trainingType, optimizer_type) -> None:
        pretrain_level = "1" if trainingType == TrainingType.BASE_PRETRAIN else "2"        
        logging.info(f"{trainingType.value} pretraining in progress, please wait...")
        print(f"{trainingType.value} pretraining in progress, please wait...")

        log_step = 500
        if self.args.method == SSL_Method.SIMCLR.value:
            trainer = SimCLRTrainer(self.args, self.writer, encoder, train_loader, pretrain_level, trainingType, log_step=log_step)

        elif self.args.method == SSL_Method.DCL.value:
            trainer = SimCLRTrainerV2(self.args, self.writer, encoder, train_loader, pretrain_level, trainingType, log_step=log_step)

        elif self.args.method == SSL_Method.MYOW.value:
            trainer = get_myow_trainer(self.args, self.writer, encoder, train_loader, pretrain_level, trainingType, log_step=log_step)

        else:
            ValueError

        model = trainer.model
        optimizer = trainer.optimizer

        for epoch in range(self.args.start_epoch, epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, (epochs - self.args.start_epoch)))
            logging.info('-' * 10)

            epoch_loss = trainer.train_epoch()

            # Decay Learning Rate
            trainer.scheduler.step()

            if epoch > 0 and epoch % 20 == 0:
                save_state(self.args, model, optimizer, pretrain_level, optimizer_type)

            logging.info(f"Epoch Loss: {epoch_loss}\t lr: {trainer.scheduler.get_last_lr()}")
            logging.info('-' * 10)

            self.args.current_epoch += 1

        save_state(self.args, model, optimizer, pretrain_level, optimizer_type)


    def first_pretrain(self) -> None:
        # initialize ResNet
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        print("=> creating model '{}'".format(self.args.resnet))

        if self.args.dataset == dataset_enum.DatasetType.IMAGENET.value or self.args.dataset == dataset_enum.DatasetType.IMAGENET_LITE.value:
            train_loader = imagenet.ImageNet(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        elif self.args.dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        else:
            train_loader = get_target_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loader()    

        self.base_pretrain(encoder, train_loader, self.args.base_epochs, trainingType=TrainingType.BASE_PRETRAIN, optimizer_type=self.args.base_optimizer)


    def second_pretrain(self) -> None:
        if self.args.do_al:
            pretrain_data = load_path_loss(self.args, self.args.pretrain_path_loss_file)
            if pretrain_data is None:
                pretext = PretextTrainer(self.args, self.writer)
                pretrain_data = pretext.do_active_learning()

            loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN).get_loader()
        else:
            loader = get_target_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loader()        

        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        self.base_pretrain(encoder, loader, self.args.target_epochs, trainingType=TrainingType.TARGET_PRETRAIN, optimizer_type=self.args.target_optimizer)