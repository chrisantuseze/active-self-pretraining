import torch
import torchvision
from utils.method_enum import Method
from models.simclr.transformation import TransformsSimCLR
from models.moco.transformation import TransformsMoCo

class CIFAR10():
    def __init__(self, args) -> None:
        self.dir = args.dataset_dir + "/cifar10"
        self.method = args.method
        self.image_size = args.image_size
        self.batch_size = args.batch_size

    def get_loader(self):
        if self.method == Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.method == Method.MOCO.value:
            transforms = TransformsMoCo(self.image_size)
        
        else:
            NotImplementedError

        train_dataset = torchvision.datasets.CIFAR10(
            self.dir,
            download=True,
            transform=transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            # shuffle=(train_sampler is None),
            drop_last=True,
            # num_workers=args.workers,
            # sampler=train_sampler,
        )

        return train_loader
    
