import glob
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import shutil
import json
import random

def InfiniteSampler(n):
    """Data sampler"""

    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
    

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def get_dir(args):
    path = 'save/gan5'
    saved_model_folder = os.path.join(path, 'models')
    saved_image_folder = os.path.join(path, 'images')
    
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    return saved_model_folder, saved_image_folder


class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, dataset, transform=None, distillation=False):
        super(ImageFolder, self).__init__()
        self.dataset = dataset
        self.dir = f'datasets/{dataset}'

        self.frame = self._parse_frame2() if distillation else self._parse_frame()
        self.transform = transform

    def _parse_frame2(self):
        print(self.dir)
        img_path = glob.glob(self.dir + '/*/*')
        return img_path

    def _parse_frame(self):
        if self.dataset in ['chest_xray', 'imagenet']:
            self.dir = f'{self.dir}/train'

        print(self.dir)
        img_path = glob.glob(self.dir + '/*/*')

        # mixing the dataset with some source proxy
        # source_proxy = glob.glob(f'datasets/cifar10/train/*/*')
        # random.shuffle(source_proxy)
        # img_path.extend(source_proxy[0:500])

        random.shuffle(img_path)

        if len(img_path) >= 2000:
            img_path = img_path[0:2000]

        return img_path

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
            
        if self.transform:
            img = self.transform(img) 

        return img



# from io import BytesIO
# import lmdb
# from torch.utils.data import Dataset


# class MultiResolutionDataset(Dataset):
#     def __init__(self, path, transform, resolution=256):
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )

#         if not self.env:
#             raise IOError('Cannot open lmdb dataset', path)

#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

#         self.resolution = resolution
#         self.transform = transform

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
#             img_bytes = txn.get(key)
#             #key_asp = f'aspect_ratio-{str(index).zfill(5)}'.encode('utf-8')
#             #aspect_ratio = float(txn.get(key_asp).decode())

#         buffer = BytesIO(img_bytes)
#         img = Image.open(buffer)
#         img = self.transform(img)

#         return img

