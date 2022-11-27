import torch
from torchvision.transforms import ToTensor, Compose, Lambda
from typing import List
from PIL import Image
import random
import glob
from datautils.path_loss import PathLoss
from models.self_sup.simclr.transformation.transformations import TransformsSimCLR
from models.self_sup.simclr.transformation.dcl_transformations import TransformsDCL
from models.utils.commons import get_params
from utils.commons import load_class_names, pil_loader, save_class_names
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from datautils.dataset_enum import DatasetType
# import cv2

labels = {}
index = 0

class PretextDataLoader():
    def __init__(self, args, path_loss_list: List[PathLoss], training_type=TrainingType.ACTIVE_LEARNING, is_val=False) -> None:
        self.args = args
        self.path_loss_list = path_loss_list

        self.training_type = training_type
        self.is_val = is_val

        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    def get_loader(self):
        if self.training_type == TrainingType.AL_FINETUNING:
            data_size = len(self.path_loss_list)
            new_data_size = int(self.args.al_finetune_data_ratio * data_size)
            self.path_loss_list = self.path_loss_list[0:new_data_size]

        if self.args.method == SSL_Method.SIMCLR.value:
            transforms = TransformsSimCLR(self.image_size)

        elif self.args.method == SSL_Method.DCL.value:
            transforms = TransformsDCL(self.image_size)

        elif self.args.method == SSL_Method.MYOW.value:
            transforms = Compose([ToTensor()])

        else:
            ValueError

        dataset = PretextDataset(self.args, self.path_loss_list, transforms, self.is_val)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
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

        self.target_transform = Lambda(lambda y: torch.zeros(len(self.label_dic), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    def __len__(self):
        return len(self.pathloss_list)

    def __getitem__(self, idx):
        path = self.pathloss_list[idx].path[0]
        if self.args.target_dataset == DatasetType.CHEST_XRAY.value or self.args.target_dataset == DatasetType.IMAGENET.value:
            img = pil_loader(path)

        else:
            img = Image.open(path)

        # img = Image.fromarray(img)

        label = path.split('/')[-2]
        return self.transform.__call__(img, not self.is_val), torch.tensor(self.label_dic[label])

class MakeBatchLoader(torch.utils.data.Dataset):
    def __init__(self, args, image_size, dir, with_train, transform=None):
        self.args = args
        self.image_size = image_size
        self.dir = dir

        if with_train:
            if self.dir == "./datasets/imagenet":
                self.img_path = glob.glob(dir + '/train/*/*/*')
            else:
                self.img_path = glob.glob(dir + '/train/*/*')
        else:
            self.img_path = glob.glob(dir + '/*/*')
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.dir == "./datasets/chest_xray" or self.dir == "./datasets/imagenet":
            img = pil_loader(self.img_path[idx])

        else:
            img = Image.open(self.img_path[idx])

        # x1, x2 = self.transform.__call__(img)
        x = self.transform.__call__(img)
        path = self.img_path[idx] 

        label = path.split('/')[-2]
        save_class_names(self.args, label)

        return x, path

class RotationLoader(torch.utils.data.Dataset):
    def __init__(self, args, dir, with_train, is_train, transform=None, training_type=TrainingType.ACTIVE_LEARNING):
        self.args = args
        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

        self.dir = args.dataset_dir + dir
        self.is_train = is_train

        if with_train:
            if self.dir == "./datasets/imagenet":
                self.img_path = glob.glob(dir + '/train/*/*/*')
            else:
                self.img_path = glob.glob(dir + '/train/*/*')
        else:
            self.img_path = glob.glob(dir + '/*/*')
        self.transform = transform

        print(self.dir)
        print(self.img_path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.dir == "./datasets/chest_xray" or self.dir == "./datasets/imagenet":
            img = pil_loader(self.img_path[idx])

        else:
            img = Image.open(self.img_path[idx])

        path = self.img_path[idx] 

        label = path.split('/')[-2]
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
            img = self.transform.__call__(img, is_train=False)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0, 1, 2, 3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], path

class Loader(torch.utils.data.Dataset):
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

        self.target_transform = Lambda(lambda y: torch.zeros(len(self.label_dic), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    
    def __len__(self):
        return len(self.pathloss_list)

    def __getitem__(self, idx):
        path = self.pathloss_list[idx].path[0]
        if self.args.target_dataset == DatasetType.CHEST_XRAY.value or self.args.target_dataset == DatasetType.IMAGENET.value:
            img = pil_loader(path)

        else:
            img = Image.open(path)

        label = path.split('/')[-2]
        return self.transform.__call__(img, not self.is_val), torch.tensor(self.label_dic[label])