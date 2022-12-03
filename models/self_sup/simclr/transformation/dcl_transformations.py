'''
Adapted from 

cite the pytorch impl of SimCLR

with little modifications
'''


import torchvision.transforms as transforms

class TransformsDCL():
    def __init__(self, size):

        color_jitter = transforms.ColorJitter(
           0.4, 0.4, 0.4, 0.1
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
            ]
        )


        self.test_transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )

    def __call__(self, x, is_train=True):
        if not is_train:
            return self.test_transform(x)

        return self.train_transform(x)
        