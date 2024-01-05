from datautils.dataset_enum import get_dataset_enum
from models.active_learning.pretext_trainer import PretextTrainer
from models.cgan.train3 import generate_dataset
from models.self_sup.swav import SwAVTrainer
from models.trainers.trainer import Trainer
from models.utils.commons import get_ds_num_classes, get_params
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

        num_classes, dir = get_ds_num_classes(self.args.source_dataset)
        model = resnet_backbone(self.args, num_classes, is_source=True, pretrained=True)
        print("=> creating model '{}'".format(self.args.backbone))

        train_loader, val_loader = get_pretrain_ds(self.args, training_type=TrainingType.SOURCE_PRETRAIN).get_loaders() 

        train_params.name = f'source_{get_dataset_enum(self.args.source_dataset)}'

        # trainer =  Trainer(self.args, self.writer, model, train_loader, val_loader, train_params)
        trainer = SwAVTrainer(self.args, model, train_loader, train_params, TrainingType.SOURCE_PRETRAIN) # Delete this if swav is longer needed
        trainer.train()

    def generate_data(self):
        generate_dataset(self.args)

    def train_target(self):
        pretext = PretextTrainer(self.args, self.writer)
        pretext.do_self_learning()