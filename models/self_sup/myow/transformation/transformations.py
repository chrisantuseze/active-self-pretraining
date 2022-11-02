'''
Adapted from 

myow

with little modifications
'''


import torchvision.transforms as transforms

class TransformsMYOW():
    
    def __init__(self, size):
        self.transform = transforms.Compose([transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        self.transform_m = transforms.Compose([transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    def __call__(self, x):
        None
        # return self.train_transform(x), self.train_transform(x)