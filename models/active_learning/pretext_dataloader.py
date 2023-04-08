from enum import Enum
import torch
import torchvision.transforms as transforms
from typing import List
from PIL import Image
import random
import glob
from datautils.path_loss import PathLoss
from models.self_sup.simclr.transformation.simclr_transformations import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from models.self_sup.swav.transformation.multicropdataset import PILRandomGaussianBlur, get_color_distortion
from models.utils.commons import get_images_pathlist, get_params
from models.utils.transformations import Transforms
from utils.commons import load_class_names, pil_loader, save_class_names
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from datautils.dataset_enum import DatasetType, get_dataset_enum
# import cv2
import utils.logger as logging

labels = {}
index = 0

class PretextDataLoader():
    def __init__(self, args, path_loss_list: List[PathLoss], training_type=TrainingType.ACTIVE_LEARNING, is_val=False, batch_size=None) -> None:
        self.args = args
        self.path_loss_list = path_loss_list

        self.training_type = training_type
        self.is_val = is_val

        self.dir = self.args.dataset_dir + "/" + get_dataset_enum(self.args.target_dataset)

        # This is done to ensure that the dataset used for validation is only a subset of the entire datasets used for training
        if is_val:
            val_path_loss_list = []

            if self.args.target_dataset in [DatasetType.IMAGENET.value, DatasetType.CHEST_XRAY.value]:
                img_paths = glob.glob(self.dir + '/train/*/*')# img_paths = glob.glob(self.dir + '/train/*/*/*')
            
            elif self.args.target_dataset == DatasetType.CIFAR10.value:
                img_paths = glob.glob(self.args.dataset_dir + '/cifar10v2/train/*/*')

            elif self.args.target_dataset in [DatasetType.UCMERCED.value, DatasetType.FOOD101.value]:
                img_paths = glob.glob(self.dir + '/images/*/*')

            elif self.args.target_dataset == DatasetType.MODERN_OFFICE_31.value:
                img_paths = glob.glob(self.dir + '/*/*/*')
            
            else:
                img_paths = glob.glob(self.dir + '/*/*')

            for path in img_paths[0:len(path_loss_list)]:
                val_path_loss_list.append(PathLoss(path, 0))     

            self.path_loss_list = val_path_loss_list 

        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size if not batch_size else batch_size

    def get_loader(self):

        # this handles the 2nd pretraining (after AL)
        if self.args.method == SSL_Method.SWAV.value and (self.training_type is not TrainingType.ACTIVE_LEARNING or self.training_type is not TrainingType.BASE_PRETRAIN):
            dataset = PretextMultiCropDataset(
                self.args,
                self.path_loss_list,
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.args.workers,
                pin_memory=True,
                # drop_last=True
            )

        else:
            if self.training_type == TrainingType.ACTIVE_LEARNING:
                transforms = Transforms(self.image_size)

            else:
                if self.args.method == SSL_Method.SIMCLR.value:
                    transforms = TransformsSimCLR(self.image_size)

                elif self.args.method == SSL_Method.DCL.value:
                    transforms = TransformsDCL(self.image_size)

                elif self.args.method == SSL_Method.MYOW.value:
                    transforms = transforms.Compose([transforms.ToTensor()])

                else:
                    ValueError

            dataset = PretextDataset(self.args, self.path_loss_list, transforms, self.is_val)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=not self.is_val,
                num_workers=self.args.workers,
                pin_memory=True,
            )

        print(f"The size of the dataset is {len(dataset)} and the number of batches is {loader.__len__()} for a batch size of {self.batch_size}")
        return loader


class PretextDataset(torch.utils.data.Dataset):
    def __init__(self, args, pathloss_list: List[PathLoss], transform, is_val=False) -> None:
        self.args = args
        self.pathloss_list = pathloss_list
        self.transform = transform
        self.is_val = is_val

        labels = set(load_class_names(self.args))
        index = 0
        self.label_dic = {}
        for label in labels:
            label = label.replace("\n", "")
            if label not in self.label_dic:
                self.label_dic[label] = index
                index += 1

    def __len__(self):
        return len(self.pathloss_list)

    def __getitem__(self, idx):
        path_loss = self.pathloss_list[idx]

        if isinstance(path_loss.path, tuple) or isinstance(path_loss.path, list):
            path = path_loss.path[0]
        else:
            path = path_loss.path

        if self.args.target_dataset in [DatasetType.CHEST_XRAY.value, DatasetType.IMAGENET.value, DatasetType.MODERN_OFFICE_31.value]:
            img = pil_loader(path)
        else:
            img = Image.open(path)

        if self.args.target_dataset == DatasetType.IMAGENET.value:
            label = path.split('/')[-2]
        else:
            label = path.split('/')[-2]

        return self.transform.__call__(img, not self.is_val), torch.tensor(self.label_dic[label])

class PretextMultiCropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        pathloss_list: List[PathLoss]=None,
    ):
        assert len(args.size_crops) == len(args.nmb_crops)
        assert len(args.min_scale_crops) == len(args.nmb_crops)
        assert len(args.max_scale_crops) == len(args.nmb_crops)

        self.args = args
        self.pathloss_list = pathloss_list

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(args.size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                args.size_crops[i],
                scale=(args.min_scale_crops[i], args.max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * args.nmb_crops[i])
        self.trans = trans

    def __len__(self):
        return len(self.pathloss_list)

    def __getitem__(self, index):
        path_loss = self.pathloss_list[index]
        if isinstance(path_loss.path, tuple) or isinstance(path_loss.path, list):
            path = path_loss.path[0]
        else:
            path = path_loss.path

        if self.args.target_dataset in [DatasetType.CHEST_XRAY.value, DatasetType.IMAGENET.value, DatasetType.MODERN_OFFICE_31.value]:
            image = pil_loader(path)
        else:
            image = Image.open(path)

        multi_crops = list(map(lambda trans: trans(image), self.trans))
        return multi_crops #TODO: Check the len of this multi_crops. Also check if you can use a mined view and an aug view here instead of just aug views.


class MakeBatchDataset(torch.utils.data.Dataset):
    def __init__(self, args, dir, with_train, is_train, is_tsne=False, transform=None, path_list=None):
        self.args = args
        params = get_params(args, TrainingType.ACTIVE_LEARNING)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

        self.dir = dir
        # self.dir = args.dataset_dir + dir

        self.is_train = is_train
        self.is_tnse = is_tsne

        self.img_path = path_list if path_list is not None else get_images_pathlist(self.dir, with_train)

        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.dir in ["./datasets/chest_xray", "./datasets/imagenet", "./datasets/food", "./datasets/modern_office_31"]:
            img = pil_loader(self.img_path[idx])
        else:
            img = Image.open(self.img_path[idx])

        path = self.img_path[idx] 
        if self.dir == "./datasets/imagenet":
            label = path.split('/')[-2]# label = path.split('/')[-3]
        else:
            label = path.split('/')[-2]

        if self.is_tnse:
            return img, label
        
        save_class_names(self.args, label)
        
        if self.is_train:
            img = self.transform.__call__(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0, 1, 2, 3]
            random.shuffle(rotations)

            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform.__call__(img, False)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0, 1, 2, 3]
            random.shuffle(rotations)

            # print(imgs[rotations[0]].shape, imgs[rotations[1]].shape, imgs[rotations[2]].shape, imgs[rotations[3]].shape)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], path