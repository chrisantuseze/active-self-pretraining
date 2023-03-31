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
        if self.method is not SSL_Method.SWAV.value or self.training_type in [TrainingType.ACTIVE_LEARNING, TrainingType.BASE_PRETRAIN]:
            if self.training_type == TrainingType.ACTIVE_LEARNING:
                transforms = Transforms(self.image_size)
                dataset = self.get_dataset(transforms)

            elif self.training_type == TrainingType.BASE_PRETRAIN:
                img_path = glob.glob(self.dir + '/*')
                logging.info(f"Original size of generated images dataset is {len(img_path)}")

                # real_target = get_images_pathlist(f'{self.args.dataset_dir}/{dataset_enum.get_dataset_enum(self.args.target_dataset)}', with_train=True)
                # random.shuffle(real_target)

                # augment_size = len(real_target) #1000
                # logging.info(f"Augmenting {augment_size} real target images to the generated dataset")
                # img_path.extend(real_target[0:augment_size])

                # #TODO: This is only for gan1
                source_proxy = glob.glob(f'{self.args.dataset_dir}/cifar10/train/*/*')
                random.shuffle(source_proxy)

                augment_size = 500
                logging.info(f"Augmenting {augment_size} proxy source images to the generated dataset")
                img_path.extend(source_proxy[0:augment_size])

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

                elif self.method == SSL_Method.MYOW.value:
                    transforms = Compose([ToTensor()])

                elif self.method == SSL_Method.SUPERVISED.value:
                    transforms = Transforms(self.image_size)

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
    # comment the two if's for gradually pretraining

    # if training_type == TrainingType.BASE_AL:
    #     print("using the Generated dataset after AL")
    #     return TargetDataset(args, f"/{args.base_dataset}", TrainingType.ACTIVE_LEARNING, is_train=is_train, batch_size=batch_size)
    
    # if training_type == TrainingType.BASE_PRETRAIN:
    #     print("using the Generated dataset")
    #     return TargetDataset(args, f"/{args.base_dataset}", TrainingType.BASE_PRETRAIN, is_train=is_train, batch_size=batch_size)
        

    if args.target_dataset == dataset_enum.DatasetType.CHEST_XRAY.value:
        print("using the CHEST XRAY dataset")
        return TargetDataset(args, "/chest_xray", training_type, with_train=True, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.REAL.value:
        print("using the REAL dataset")
        return TargetDataset(args, "/real", training_type, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.UCMERCED.value:
        print("using the UCMERCED dataset")
        return TargetDataset(args, "/ucmerced/images", training_type, is_train=is_train, batch_size=batch_size)
    
    elif args.target_dataset == dataset_enum.DatasetType.IMAGENET.value:
        print("using the IMAGENET dataset")
        return TargetDataset(args, "/imagenet", training_type, with_train=True, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.CIFAR10.value:
        print("using the CIFAR10 dataset")
        return TargetDataset(args, "/cifar10v2", training_type, with_train=True, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.FLOWERS.value:
        print("using the FLOWERS dataset")
        return TargetDataset(args, "/flowers", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.EUROSAT.value:
        print("using the EUROSAT dataset")
        return TargetDataset(args, "/eurosat", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.FOOD101.value:
        print("using the FOOD101 dataset")
        return TargetDataset(args, "/food-101/images", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.CLIPART.value:
        print("using the CLIPART dataset")
        return TargetDataset(args, "/clipart", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    elif args.target_dataset == dataset_enum.DatasetType.SKETCH.value:
            print("using the SKETCH dataset")
            return TargetDataset(args, "/sketch", training_type, with_train=False, is_train=is_train, batch_size=batch_size)

    else:
        ValueError