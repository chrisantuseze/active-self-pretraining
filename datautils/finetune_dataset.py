import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Finetune():
    def __init__(self, args) -> None:
        self.args = args
        self.dir = args.dataset_dir + "/finetune"
        self.image_size = args.image_size
        self.batch_size = args.batch_size

    def get_loader(self):
        traindir = os.path.join(self.dir, 'train')
        valdir = os.path.join(self.dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=self.args.batch_size,
        num_workers=self.args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)


        return train_loader, val_loader
    
