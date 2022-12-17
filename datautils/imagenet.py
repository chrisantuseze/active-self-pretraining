from random import shuffle
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from torch.utils.data import random_split
from models.self_sup.swav.transformation.swav_transformation import TransformsSwAV

from models.utils.commons import get_params, split_dataset
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from models.self_sup.simclr.transformation.simclr_transformations import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from datautils.dataset_enum import DatasetType
from models.utils.transformations import Transforms


class ImageNet():
    def __init__(self, args, training_type=TrainingType.BASE_PRETRAIN) -> None:
        self.dir = args.dataset_dir + "/imagenet"
        self.method = args.method
        self.args = args
        
        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    def get_loader(self):
        if self.method is not SSL_Method.SWAV.value:
            if self.method == SSL_Method.SIMCLR.value:
                transforms = TransformsSimCLR(self.image_size)

            elif self.method == SSL_Method.DCL.value:
                transforms = TransformsDCL(self.image_size)

            elif self.method == SSL_Method.MYOW.value:
                transforms = Compose([ToTensor()])

            elif self.method == SSL_Method.SUPERVISED.value:
                transforms = Transforms(self.image_size)

            else:
                NotImplementedError

            train_ds, val_ds = split_dataset(self.args, self.dir, transforms)

            loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=self.batch_size,
                drop_last=True, 
                shuffle=True
            )
        
        else:
            swav = TransformsSwAV(self.args, self.batch_size, self.dir)
            loader, train_ds = swav.train_loader, swav.train_dataset

        print(f"The size of the ImageNet dataset is {len(train_ds)} and the number of batches is ", loader.__len__())

        return loader
    
