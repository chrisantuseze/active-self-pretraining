import glob
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
import random

from datautils.path_loss import PathLoss

from models.active_learning.pretext_dataloader import MakeBatchDataset, PretextMultiCropDataset
from models.self_sup.simclr.transformation import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from models.self_sup.swav.transformation.swav_transformation import TransformsSwAV
from models.utils.commons import get_images_pathlist, get_params, split_dataset2
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method

from datautils import dataset_enum
from models.utils.transformations import Transforms

import utils.logger as logging

class TargetDataset():
    def __init__(self, args, dir, training_type=TrainingType.BASE_PRETRAIN, with_train=False, is_train=True, batch_size=None) -> None:
        self.args = args
        self.dir = args.dataset_dir + dir
        self.method = args.method
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

    def get_finetuner_loaders(self, train_batch_size, val_batch_size, path_list=None):
        transforms = Transforms(self.image_size)
        dataset = MakeBatchDataset(
            self.args, self.dir, self.with_train, self.is_train, 
            is_tsne=False, transform=transforms, path_list=path_list)

        train_ds, val_ds = split_dataset2(dataset=dataset, ratio=0.7, is_classifier=True)

        train_loader = torch.utils.data.DataLoader(
                    train_ds, 
                    batch_size=train_batch_size,
                    num_workers=self.args.workers,
                    shuffle=True,
                    pin_memory=True
                )
        val_loader = torch.utils.data.DataLoader(
                        val_ds, 
                        batch_size=val_batch_size, 
                        num_workers=self.args.workers,
                        shuffle=False,
                        pin_memory=True
                    )

        print(f"The size of the dataset is ({len(train_ds)}, {len(val_ds)}) and the number of batches is ({train_loader.__len__()}, {val_loader.__len__()}) for a batch size of {self.batch_size}")

        return train_loader, val_loader

    def get_loader(self):
        if self.method is not SSL_Method.SWAV.value or self.training_type in [TrainingType.ACTIVE_LEARNING] or (self.args.training_type == "single_iter" and TrainingType.TARGET_PRETRAIN) or (self.args.training_type == "proxy_source" and TrainingType.BASE_PRETRAIN):
            if self.training_type == TrainingType.ACTIVE_LEARNING:
                transforms = Transforms(self.image_size)
                dataset = self.get_dataset(transforms)

            elif self.args.training_type == "single_iter" and self.training_type == TrainingType.TARGET_PRETRAIN:
                img_path = get_images_pathlist(f'{self.args.dataset_dir}/{self.args.base_dataset}', with_train=True)
                logging.info(f"Original size of generated images dataset is {len(img_path)}")

                real_target = get_images_pathlist(f'{self.args.dataset_dir}/{dataset_enum.get_dataset_enum(self.args.target_dataset)}', with_train=False)
                random.shuffle(real_target)
                img_path.extend(real_target)

                logging.info(f"Total size of dataset is {len(img_path)}")
                
                path_loss_list = [PathLoss(path, 0) for path in img_path]
                
                dataset = PretextMultiCropDataset(
                    self.args,
                    path_loss_list,
                )

            elif self.args.training_type == "proxy_source" and self.training_type == TrainingType.BASE_PRETRAIN:
                img_path = glob.glob(self.dir + '/train/*/*')

                path_loss_list = [PathLoss(path, 0) for path in img_path]
                
                dataset = PretextMultiCropDataset(
                    self.args,
                    path_loss_list,
                )

            else:
                if self.method == SSL_Method.SIMCLR.value:
                    transforms = TransformsSimCLR(self.image_size)

                if self.method == SSL_Method.DCL.value:
                    transforms = TransformsDCL(self.image_size)

                else:
                    ValueError

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
    

def get_target_pretrain_ds(args, training_type=TrainingType.BASE_PRETRAIN, is_train=True, batch_size=None) -> TargetDataset:

    if args.training_type == "proxy_source" and training_type == TrainingType.BASE_PRETRAIN:
        print("using the proxy dataset")
        return TargetDataset(args, "/cifar10", TrainingType.BASE_PRETRAIN, is_train=is_train, batch_size=batch_size)

    if args.target_dataset == dataset_enum.DatasetType.CHEST_XRAY.value:
        print("using the CHEST XRAY dataset")
        return TargetDataset(args, "/chest_xray", training_type, with_train=True, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.FLOWERS.value:
        print("using the FLOWERS dataset")
        return TargetDataset(args, "/flowers", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.EUROSAT.value:
        print("using the EUROSAT dataset")
        return TargetDataset(args, "/eurosat", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.HAM10000.value:
        print("using the HAM10000 dataset")
        return TargetDataset(args, "/ham10000", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.CLIPART.value:
        print("using the CLIPART dataset")
        return TargetDataset(args, "/clipart", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.SKETCH.value:
        print("using the SKETCH dataset")
        return TargetDataset(args, "/sketch", training_type, with_train=False, is_train=is_train, batch_size=batch_size)
    
    elif args.target_dataset == dataset_enum.DatasetType.QUICKDRAW.value:
        print("using the QUICKDRAW dataset")
        return TargetDataset(args, "/quickdraw", training_type, with_train=False, is_train=is_train, batch_size=batch_size)
    
    elif args.target_dataset == dataset_enum.DatasetType.MODERN_OFFICE_31.value:
        print("using the MODERN_OFFICE_31 dataset")
        return TargetDataset(args, "/modern_office_31", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.AMAZON.value:
        print("using the Office-31 AMAZON dataset")
        return TargetDataset(args, "/amazon/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.WEBCAM.value:
        print("using the Office-31 WEBCAM dataset")
        return TargetDataset(args, "/webcam/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.DSLR.value:
        print("using the Office-31 DSLR dataset")
        return TargetDataset(args, "/dslr/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.PAINTING.value:
        print("using the PAINTING dataset")
        return TargetDataset(args, "/painting", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.ARTISTIC.value:
        print("using the OfficeHome ARTISTIC dataset")
        return TargetDataset(args, "/artistic", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.CLIP_ART.value:
        print("using the OfficeHome CLIP_ART dataset")
        return TargetDataset(args, "/clip_art", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.PRODUCT.value:
        print("using the OfficeHome PRODUCT dataset")
        return TargetDataset(args, "/product", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.REAL_WORLD.value:
        print("using the OfficeHome REAL_WORLD dataset")
        return TargetDataset(args, "/real_world", training_type, with_train=False, is_train=is_train, batch_size=batch_size)
    
    else:
        ValueError