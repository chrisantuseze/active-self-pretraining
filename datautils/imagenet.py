import torch
import torchvision
from utils.method_enum import Method
from models.methods.simclr.transformation import TransformsSimCLR
from models.methods.moco.transformation import TransformsMoCo

class ImageNet():
    def __init__(self, args) -> None:
        self.dir = args.dataset_dir + "/imagenet"
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

        dataset = torchvision.datasets.ImageFolder(
            self.dir,
            transform=transforms)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
        )

        return loader
    
