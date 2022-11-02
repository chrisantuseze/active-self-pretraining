import torch
import torchvision
from torchvision.transforms import ToTensor, Compose

from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from utils.method_enum import Method
from models.self_sup.simclr.transformation import TransformsSimCLR

class CIFAR10():
    def __init__(self, args, training_type=TrainingType.BASE_PRETRAIN) -> None:
        self.dir = args.dataset_dir + "/cifar10"
        self.method = args.method

        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    def get_loader(self):
        if self.method == Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.method == Method.MYOW.value:
            transforms = Compose([ToTensor()])

        else:
            NotImplementedError

        dataset = torchvision.datasets.CIFAR10(
            self.dir,
            download=True,
            transform=transforms)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
        )

        print(f"The size of the Cifar10 dataset is {len(dataset)} and the number of batches is ", loader.__len__())

        return loader
    
