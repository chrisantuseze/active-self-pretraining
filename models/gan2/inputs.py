import glob
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from models.gan.dataloaders.ImageListDataset import ImageListDataset

def get_dataset(name, data_dir, size=64, lsun_categories=None):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])

    if name == 'image':
        # dataset = datasets.ImageFolder(data_dir, transform)

        img_path_list = glob.glob(data_dir + "/*.png")
        img_path_list = [[path, i] for i, path in enumerate(sorted(img_path_list))]
        dataset = ImageListDataset(img_path_list, transform=transform)

        nlabels = 1 #len(dataset.classes)
        
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=False,
                                   transform=transform)
        nlabels = 10
    else:
        raise NotImplemented

    return dataset, nlabels


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
