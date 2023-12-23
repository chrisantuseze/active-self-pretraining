import torchvision.transforms as transforms

class Transforms():
    def __init__(self, size, is_train=True):
        self.is_train = is_train

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, x):
        if not self.is_train:
            return self.test_transform(x)

        return self.train_transform(x)