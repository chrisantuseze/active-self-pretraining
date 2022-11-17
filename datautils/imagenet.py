import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from torch.utils.data import random_split

from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from models.self_sup.simclr.transformation.transformations import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from datautils.dataset_enum import DatasetType


class ImageNet():
    def __init__(self, args, training_type=TrainingType.BASE_PRETRAIN) -> None:
        self.dir = args.dataset_dir + "/imagenet"
        self.method = args.method
        self.args = args
        
        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    def split_dataset(self, transforms):
        dataset = datasets.ImageFolder(
            self.dir,
            transform=transforms)

        ratio = 1.0 if self.args.dataset == DatasetType.IMAGENET.value else 0.6
        train_size = int(ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_ds, val_ds = random_split(dataset=dataset, lengths=[train_size, val_size])
        return train_ds

    def get_loader(self):
        if self.method == SSL_Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.method == SSL_Method.DCL.value:
            transforms = TransformsDCL(self.image_size)

        elif self.method == SSL_Method.MYOW.value:
            transforms = Compose([ToTensor()])

        else:
            NotImplementedError

        dataset = self.split_dataset(transforms)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True, 
        )

        print(f"The size of the ImageNet dataset is {len(dataset)} and the number of batches is ", loader.__len__())

        return loader
    
