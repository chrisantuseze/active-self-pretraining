import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datautils.dataset_enum import DatasetType
from models.active_learning.pretext_dataloader import PretextDataLoader

from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType

class Finetune():
    def __init__(self, args, dir, training_type=TrainingType.BASE_PRETRAIN) -> None:
        self.args = args
        self.dir = args.dataset_dir + dir
        
        params = get_params(args, training_type)
        self.batch_size = params.batch_size


    def split_dataset(self, normalize):
        dataset = datasets.ImageFolder(
            self.dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_ds, val_ds = random_split(dataset=dataset, lengths=[train_size, val_size])
        return train_ds, val_ds

    def get_loader(self, pretrain_data=None):

        if pretrain_data:
            train_loader = PretextDataLoader(
                self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN, 
                is_val=False).get_loader()

            val_loader = PretextDataLoader(
                self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN, 
                is_val=True).get_loader()

            return train_loader, val_loader 

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

        val_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

        if self.args.finetune_dataset == DatasetType.IMAGENET.value or self.args.finetune_dataset == DatasetType.IMAGENET_LITE.value:
            traindir = os.path.join(self.dir, 'train')
            valdir = os.path.join(self.dir, 'val')

            train_dataset = datasets.ImageFolder(
                traindir,
                transform=train_transforms
                )

            val_dataset = datasets.ImageFolder(
                valdir,
                transform=val_transforms
                )

        elif self.args.finetune_dataset == DatasetType.CIFAR10.value:
            train_dataset = torchvision.datasets.CIFAR10(
                self.dir,
                train=True,
                download=True,
                transform=train_transforms)

            val_dataset = torchvision.datasets.CIFAR10(
                self.dir,
                train=False,
                download=True,
                transform=val_transforms)

        else:
            train_dataset, val_dataset = self.split_dataset(normalize)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True
                        )

        val_loader = torch.utils.data.DataLoader(
                        val_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False,
                        pin_memory=True
                    )

        return train_loader, val_loader
    
