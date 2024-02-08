import torch
import torchvision.transforms as transforms

from typing import List
from PIL import Image
import random
from datautils.path_loss import PathLoss
from models.trainers.transformation.multicropdataset import PILRandomGaussianBlur, get_color_distortion
from models.utils.commons import get_images_pathlist, get_params
from models.utils.training_type_enum import TrainingType
from datautils.dataset_enum import get_dataset_info

class PretextDataLoader():
    def __init__(self, args, path_loss_list: List[PathLoss], training_type=TrainingType.ACTIVE_LEARNING, is_val=False, batch_size=None) -> None:
        self.args = args
        self.path_loss_list = path_loss_list

        self.training_type = training_type
        self.is_val = is_val

        self.dir = self.args.dataset_dir + "/" + get_dataset_info(self.args.target_dataset)[2]

        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size if not batch_size else batch_size

    def get_loader(self):
        dataset = PretextMultiCropDataset(self.args, self.path_loss_list)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        print(f"The size of the dataset is {len(dataset)} and the number of batches is {loader.__len__()} for a batch size of {self.batch_size}")
        return loader


# @DeprecationWarning("This has been deprecated")
class PretextDataset(torch.utils.data.Dataset):
    def __init__(self, args, pathloss_list: List[PathLoss], transform) -> None:
        self.args = args
        self.pathloss_list = pathloss_list
        self.transform = transform

    def __len__(self):
        return len(self.pathloss_list)

    def __getitem__(self, idx):
        path_loss = self.pathloss_list[idx]

        if isinstance(path_loss.path, tuple) or isinstance(path_loss.path, list):
            path = path_loss.path[0]
        else:
            path = path_loss.path

        img = Image.open(path)
        image = self.transform(img)

        return image, torch.tensor(path_loss.label)

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

        image = Image.open(path)

        multi_crops = list(map(lambda trans: trans(image), self.trans))
        return multi_crops


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
        img = Image.open(self.img_path[idx])

        path = self.img_path[idx] 
        label = path.split('/')[-2]

        if self.is_tnse:
            return self.transform(img), torch.tensor(0)
                
        img = self.transform(img)
        img1 = torch.rot90(img, 1, [1,2])
        img2 = torch.rot90(img, 2, [1,2])
        img3 = torch.rot90(img, 3, [1,2])
        imgs = [img, img1, img2, img3]
        rotations = [0, 1, 2, 3]
        random.shuffle(rotations)
        if self.is_train:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], path