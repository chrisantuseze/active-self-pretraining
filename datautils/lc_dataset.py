import torch
import torchvision.transforms as transforms
import torchvision

from models.utils.commons import get_params, split_dataset
from models.utils.training_type_enum import TrainingType

class LCDataset():
    def __init__(self, args, dir, training_type=TrainingType.BASE_PRETRAIN) -> None:
        self.args = args
        self.dir = args.dataset_dir + dir
        params = get_params(args, training_type)
        self.batch_size = params.batch_size

    def split_dataset(self, normalize):
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        return split_dataset(self.args, self.dir, transform, ratio=0.8, is_classifier=True)

    def get_loader(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
        # train_dataset, val_dataset = self.split_dataset(normalize)

        train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAutocontrast(0.5),
                transforms.ToTensor(),
                normalize
        ])  
        train_dataset = torchvision.datasets.ImageFolder(self.dir, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size,
                num_workers=self.args.workers,
                shuffle=True, pin_memory=True, drop_last=True
        )
        
        val_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
        ])
        val_dataset = torchvision.datasets.ImageFolder(self.dir, transform=val_transform)        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, 
            num_workers=self.args.workers, shuffle=False
        )

        print(f"The size of the dataset is ({len(train_dataset)}, {len(val_dataset)}) and the number of batches is ({train_loader.__len__()}, {val_loader.__len__()}) for a batch size of {self.batch_size}")
        return train_loader, val_loader
    
