import torch
import torchvision


from models.active_learning.pretext_dataloader import MakeBatchDataset
from models.trainers.transformation.swav_transformation import TransformsSwAV
from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType

from datautils import dataset_enum
from models.utils.transformations import Transforms, get_train_val_transforms

import utils.logger as logging

class TargetDataset():
    def __init__(self, args, dir, training_type=TrainingType.BASE_PRETRAIN, with_train=False, is_train=True, batch_size=None) -> None:
        self.args = args
        self.dir = args.dataset_dir + dir
        self.training_type = training_type
        self.with_train = with_train
        self.is_train = is_train
        
        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size if not batch_size else batch_size

    
    def get_dataset(self, transforms, is_tsne=False):
        return MakeBatchDataset(
            self.args, self.dir, self.with_train, 
            self.is_train, is_tsne, transforms) if self.training_type == TrainingType.ACTIVE_LEARNING else torchvision.datasets.ImageFolder(
                                                                                                self.dir, transform=transforms)

    def get_finetuner_loaders(self, path_list=None):
        train_transform, val_transform = get_train_val_transforms()
        
        train_dataset = MakeBatchDataset(
            self.args, self.dir, self.with_train, self.is_train, 
            is_tsne=False, transform=train_transform, path_list=path_list)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            num_workers=self.args.workers,
            shuffle=True, pin_memory=True, drop_last=True
        )
        
        val_dataset = MakeBatchDataset(
            self.args, self.dir, self.with_train, self.is_train, 
            is_tsne=False, transform=val_transform, path_list=path_list)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, 
            num_workers=self.args.workers, shuffle=False
        )

        print(f"The size of the dataset is ({len(train_dataset)}, {len(val_dataset)}) and the number of batches is ({train_loader.__len__()}, {val_loader.__len__()}) for a batch size of {self.batch_size}")
        return train_loader, val_loader

    def get_loader(self):
        if self.training_type == TrainingType.ACTIVE_LEARNING:
            transforms = Transforms(self.image_size)
            dataset = self.get_dataset(transforms)

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=self.is_train, 
                num_workers=self.args.workers
            )
        
        else:
            swav = TransformsSwAV(self.args, self.batch_size, self.dir)
            loader, dataset = swav.train_loader, swav.train_dataset
        
        logging.info(f"The size of the dataset is {len(dataset)} and the number of batches is {loader.__len__()} for a batch size of {self.batch_size}")

        return loader
    

def get_pretrain_ds(args, training_type=TrainingType.BASE_PRETRAIN, is_train=True, batch_size=None) -> TargetDataset:
    if training_type == TrainingType.BASE_PRETRAIN:
        dataset_type = args.base_dataset
    else:
        dataset_type = args.target_dataset

    if dataset_type == dataset_enum.DatasetType.CLIPART.value:
        print("using the CLIPART dataset")
        return TargetDataset(args, "/clipart", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.SKETCH.value:
        print("using the SKETCH dataset")
        return TargetDataset(args, "/sketch", training_type, with_train=False, is_train=is_train, batch_size=batch_size)
    
    elif dataset_type == dataset_enum.DatasetType.QUICKDRAW.value:
        print("using the QUICKDRAW dataset")
        return TargetDataset(args, "/quickdraw", training_type, with_train=False, is_train=is_train, batch_size=batch_size)
    
    elif dataset_type == dataset_enum.DatasetType.AMAZON.value:
        print("using the Office-31 AMAZON dataset")
        return TargetDataset(args, "/office-31/amazon/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.WEBCAM.value:
        print("using the Office-31 WEBCAM dataset")
        return TargetDataset(args, "/office-31/webcam/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.DSLR.value:
        print("using the Office-31 DSLR dataset")
        return TargetDataset(args, "/office-31/dslr/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.PAINTING.value:
        print("using the PAINTING dataset")
        return TargetDataset(args, "/painting", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.ARTISTIC.value:
        print("using the OfficeHome ARTISTIC dataset")
        return TargetDataset(args, "/officehome/artistic", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.CLIP_ART.value:
        print("using the OfficeHome CLIP_ART dataset")
        return TargetDataset(args, "/officehome/clip_art", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.PRODUCT.value:
        print("using the OfficeHome PRODUCT dataset")
        return TargetDataset(args, "/officehome/product", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif dataset_type == dataset_enum.DatasetType.REAL_WORLD.value:
        print("using the OfficeHome REAL_WORLD dataset")
        return TargetDataset(args, "/officehome/real_world", training_type, with_train=False, is_train=is_train, batch_size=batch_size)
    
    else:
        ValueError