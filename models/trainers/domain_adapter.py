from datautils.dataset_enum import get_dataset_enum
from models.active_learning.pretext_trainer import PretextTrainer
from models.cgan.train3 import train
from models.trainers.trainer import Trainer
from models.utils.commons import get_params
import torch
import torch.nn as nn

from datautils.target_dataset import get_pretrain_ds
from models.trainers.resnet import resnet_backbone
import utils.logger as logging
from models.utils.training_type_enum import TrainingType

class DomainAdapter:
    def __init__(self, args, writer) -> None:
        self.args = args
        self.writer = writer
        logging.info(f"Source = {get_dataset_enum(self.args.source_dataset)}, Target = {get_dataset_enum(self.args.target_dataset)}")

    def train_source(self):
        train_params = get_params(self.args, TrainingType.SOURCE_PRETRAIN)

        model = resnet_backbone(self.args.backbone, pretrained=False)
        print("=> creating model '{}'".format(self.args.backbone))

        train_loader, val_loader = get_pretrain_ds(self.args, training_type=TrainingType.SOURCE_PRETRAIN).get_loaders() 

        train_params.name = f'source_{self.args.source_dataset}'

        trainer = Trainer(self.args, self.writer, model, train_loader, val_loader, train_params)
        trainer.train()

    def generate_data(self):
        train(self.args)

    def train_target(self):
        pretext = PretextTrainer(self.args, self.writer)
        pretext.do_self_learning()