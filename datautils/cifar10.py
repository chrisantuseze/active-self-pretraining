import torch
import torchvision
from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
from utils.method_enum import Method
from models.methods.simclr.transformation import TransformsSimCLR
from models.methods.moco.transformation.transformations import TransformsMoCo

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

        elif self.method == Method.MOCO.value:
            transforms = TransformsMoCo(self.image_size)

        elif self.method == Method.SWAV.value:
            NotImplementedError
        
        else:
            NotImplementedError

        dataset = torchvision.datasets.CIFAR10(
            self.dir,
            download=True,
            transform=transforms)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            # shuffle=(train_sampler is None),
            drop_last=True,
            # num_workers=args.workers,
            # sampler=train_sampler,
        )

        return loader
    
