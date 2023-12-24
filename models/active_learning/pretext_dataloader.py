import torch

from typing import List
from PIL import Image
import random
from datautils.path_loss import PathLoss
from models.utils.commons import get_images_pathlist, get_params
from models.utils.transformations import Transforms
from utils.commons import load_class_names, pil_loader, save_class_names
from models.utils.training_type_enum import TrainingType
# import cv2

labels = {}
index = 0

class PretextDataLoader():
    def __init__(self, args, path_loss_list: List[PathLoss], training_type=TrainingType.ACTIVE_LEARNING, is_val=False, batch_size=None) -> None:
        self.args = args
        self.path_loss_list = path_loss_list

        self.training_type = training_type
        self.is_val = is_val

        # This is done to ensure that the dataset used for validation is only a subset of the entire datasets used for training
        # NO LONGER NEEDED
        # self.dir = self.args.dataset_dir + "/" + get_dataset_enum(self.args.target_dataset)
        if is_val:
            # val_path_loss_list = []

            # if self.args.target_dataset in [DatasetType.AMAZON.value, DatasetType.DSLR.value, DatasetType.WEBCAM.value]:
            #     img_paths = glob.glob(self.dir + '/images/*/*')

            # else:
            #     img_paths = glob.glob(self.dir + '/*/*')

            # for path in img_paths[0:len(path_loss_list)]:
            #     val_path_loss_list.append(PathLoss(path, 0))     

            # self.path_loss_list = val_path_loss_list 
            pass

        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size if not batch_size else batch_size

    def get_loader(self):
        dataset = PretextDataset(self.args, self.path_loss_list, self.image_size, self.is_val)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=not self.is_val,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        print(f"The size of the dataset is {len(dataset)} and the number of batches is {loader.__len__()} for a batch size of {self.batch_size}")
        return loader
    
    def get_loaders(self):
        split_index = int(len(self.path_loss_list)* 0.9)

        train = self.path_loss_list[:split_index]
        val = self.path_loss_list[split_index:]

        train_dataset = PretextDataset(self.args, train, self.image_size, is_val=False)
        val_dataset = PretextDataset(self.args, val, self.image_size, is_val=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        print(f"The size of the train dataset is {len(train_dataset)}, size of val is {len(val_dataset)}, and the number of batches is {train_loader.__len__()} for a batch size of {self.batch_size}")
        return train_loader, val_loader


class PretextDataset(torch.utils.data.Dataset):
    def __init__(self, args, pathloss_list: List[PathLoss], image_size, is_val=False) -> None:
        self.args = args
        self.pathloss_list = pathloss_list
        self.transform = Transforms(image_size, is_val)
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

        img = Image.open(path)

        if path_loss.label:
            label = path_loss.label
        elif self.label_dic:
            label = path.split('/')[-2]
            label = torch.tensor(self.label_dic[label])
            print("here")
        else:
            label = 0

        image = self.transform.__call__(img)

        if not isinstance(label, int):
            print(label)

        return image, label

class MakeBatchDataset(torch.utils.data.Dataset):
    def __init__(self, args, dir, with_train, is_train, is_tsne=False, transform=None, path_list=None):
        self.args = args
        params = get_params(args, TrainingType.ACTIVE_LEARNING)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

        self.dir = dir

        self.is_train = is_train
        self.is_tnse = is_tsne

        self.img_path = path_list if path_list is not None else get_images_pathlist(self.dir, with_train)

        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.dir in ["./datasets/chest_xray", "./datasets/modern_office_31"]:
            img = pil_loader(self.img_path[idx])
        else:
            img = Image.open(self.img_path[idx])

        path = self.img_path[idx] 
        label = path.split('/')[-2]

        if self.is_tnse:
            return self.transform.__call__(img), torch.tensor(0)
        
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

            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], path