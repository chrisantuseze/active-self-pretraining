import torch
import torchvision

from models.active_learning.pretext_dataloader import MakeBatchLoader
from models.methods.moco.transformation.transformations import TransformsMoCo
from models.methods.simclr.transformation import TransformsSimCLR
from utils.method_enum import Method

from datautils import dataset_enum

class TargetDataset():
    def __init__(self, args, isAL, dir) -> None:
        self.dir = args.dataset_dir + dir
        self.isAL = isAL
        self.method = args.method
        self.image_size = args.al_image_size if isAL else args.image_size
        self.batch_size = args.al_batch_size if isAL else  args.batch_size

    
    def get_dataset(self, transforms):
        return MakeBatchLoader(self.image_size, self.dir, transforms) if self.isAL else torchvision.datasets.ImageFolder(
            self.dir,
            transform=transforms)

    def get_loader(self):
        if self.method == Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.method == Method.MOCO.value:
            transforms = TransformsMoCo(self.image_size)

        elif self.method == Method.SWAV.value:
            NotImplementedError
        
        else:
            NotImplementedError

        dataset = self.get_dataset(transforms)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
        )

        return loader
    

def get_target_pretrain_ds(args, isAL=True):
    if args.target_dataset == dataset_enum.DatasetType.UCMERCED.value:
        print("using the UCMERCED dataset")
        return TargetDataset(args, isAL, "/UCMerced_LandUse")
    
    elif args.target_dataset == dataset_enum.DatasetType.SKETCH.value:
        print("using the SKETCH dataset")
        return TargetDataset(args, isAL, "/sketch")

    elif args.target_dataset == dataset_enum.DatasetType.CLIPART.value:
        print("using the CLIPART dataset")
        return TargetDataset(args, isAL, "/clipart")
    
    elif args.target_dataset == dataset_enum.DatasetType.IMAGENET.value:
        print("using the IMAGENET dataset")
        return TargetDataset(args, isAL, "/imagenet")

    else:
        NotImplementedError