from random import shuffle
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from models.self_sup.swav.transformation.swav_transformation import TransformsSwAV
from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from models.self_sup.simclr.transformation.simclr_transformations import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from models.utils.transformations import Transforms

class CIFAR10():
    def __init__(self, args, training_type=TrainingType.BASE_PRETRAIN) -> None:
        self.dir = args.dataset_dir + "/cifar10v2"
        self.method = args.method

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

            # dataset = torchvision.datasets.CIFAR10(
            #     self.dir,
            #     download=True,
            #     transform=transforms)

            dataset = datasets.ImageFolder(
                self.dir,
                transform=transforms)

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True
            )

        else:
            swav = TransformsSwAV(self.args, self.dir, self.batch_size)
            loader, dataset = swav.train_loader, swav.train_dataset

        print(f"The size of the Cifar10 dataset is {len(dataset)} and the number of batches is ", loader.__len__())

        return loader
    
