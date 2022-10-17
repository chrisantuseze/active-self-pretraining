import torch
import torchvision
from utils.method_enum import Method
from models.methods.simclr.transformation import TransformsSimCLR
from models.methods.moco.transformation import TransformsMoCo

class UCMerced():
    def __init__(self, args) -> None:
        self.dir = args.dataset_dir + "/UCMerced_LandUse"
        self.method = args.method
        self.image_size = args.image_size
        self.batch_size = args.batch_size

    def get_loader(self):
        if self.method == Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.method == Method.MOCO.value:
            transforms = TransformsMoCo(self.image_size)

        elif self.method == Method.SWAV.value:
            NotImplementedError
        
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
    
