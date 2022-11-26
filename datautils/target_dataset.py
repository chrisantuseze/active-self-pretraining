import torch
import torchvision
from torchvision.transforms import ToTensor, Compose

from models.active_learning.pretext_dataloader import MakeBatchLoader
from models.self_sup.simclr.transformation import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from models.utils.commons import get_params, split_dataset
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method

from datautils import dataset_enum

class TargetDataset():
    def __init__(self, args, dir, training_type=TrainingType.BASE_PRETRAIN, with_train=False) -> None:
        self.args = args
        self.dir = args.dataset_dir + dir
        self.method = args.method
        self.training_type = training_type
        self.with_train = with_train
        
        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    
    def get_dataset(self, transforms):
        return MakeBatchLoader(
            self.args,
            self.image_size, 
            self.dir, self.with_train, transforms) if self.training_type == TrainingType.ACTIVE_LEARNING else torchvision.datasets.ImageFolder(
                                                                                                self.dir,
                                                                                                transform=transforms)

    def get_loader(self):
        if self.method == SSL_Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        if self.method == SSL_Method.DCL.value:
            transforms = TransformsDCL(self.image_size)

        elif self.method == SSL_Method.MYOW.value:
            transforms = Compose([ToTensor()])

        else:
            ValueError

        dataset = self.get_dataset(transforms)
        print(len(dataset))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
        )
        
        print(f"The size of the dataset is {len(dataset)} and the number of batches is {loader.__len__()} for a batch size of {self.batch_size}")

        return loader
    

def get_target_pretrain_ds(args, training_type=TrainingType.BASE_PRETRAIN):
    if args.target_dataset == dataset_enum.DatasetType.QUICKDRAW.value:
        print("using the QUICKDRAW dataset")
        return TargetDataset(args, "/quickdraw", training_type)
    
    elif args.target_dataset == dataset_enum.DatasetType.SKETCH.value:
        print("using the SKETCH dataset")
        return TargetDataset(args, "/sketch", training_type)

    elif args.target_dataset == dataset_enum.DatasetType.CLIPART.value:
        print("using the CLIPART dataset")
        return TargetDataset(args, "/clipart", training_type)

    elif args.target_dataset == dataset_enum.DatasetType.UCMERCED.value:
        print("using the UCMERCED dataset")
        return TargetDataset(args, "/ucmerced/images", training_type)
    
    elif args.target_dataset == dataset_enum.DatasetType.IMAGENET.value:
        print("using the IMAGENET dataset")
        return TargetDataset(args, "/imagenet", training_type, with_train=True)

    elif args.target_dataset == dataset_enum.DatasetType.IMAGENET_LITE.value:
        print("using the IMAGENET dataset")
        return TargetDataset(args, "/imagenet", training_type, with_train=True)

    elif args.target_dataset == dataset_enum.DatasetType.CIFAR10.value:
        print("using the CIFAR10 dataset")
        return TargetDataset(args, "/cifar10v2", training_type, with_train=True)

    else:
        ValueError