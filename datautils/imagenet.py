import torch
import torchvision
from torchvision.transforms import ToTensor, Compose

from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from models.self_sup.simclr.transformation import TransformsSimCLR
from datautils.dataset_enum import DatasetType


class ImageNet():
    def __init__(self, args, training_type=TrainingType.BASE_PRETRAIN) -> None:
        dir = "/imagenet" if args.dataset == DatasetType.IMAGENET.value else "/imagenet_lite"
        self.dir = args.dataset_dir + dir
        self.method = args.method
        
        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    def get_loader(self):
        if self.method == SSL_Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.method == SSL_Method.MYOW.value:
            transforms = Compose([ToTensor()])

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

        print(f"The size of the ImageNet dataset is {len(dataset)} and the number of batches is ", loader.__len__())

        return loader
    
