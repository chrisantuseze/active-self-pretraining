import torch
from PIL import Image
import glob
from models.methods.simclr.transformation.transformations import TransformsSimCLR
from models.utils.commons import get_params
from models.utils.training_type_enum import TrainingType
# import cv2


class PretextDataset(torch.utils.data.Dataset):
    def __init__(self, img_loss_list) -> None:
        super(PretextDataset, self).__init__()
        self.img_loss_list = img_loss_list
        self.images1, self.images2 = self.get_images(self.img_loss_list)

    def get_images(self, img_loss_list):
        images1 = []
        images2 = []

        for image_loss in img_loss_list:
            images1.append(image_loss.image1)
            images2.append(image_loss.image2)

        return images1, images2

    def __len__(self):
        return len(self.img_loss_list)

    def __getitem__(self, idx):
        image1, image2 = self.images1[idx], self.images2[idx]
        return image1, image2


class PretextDataLoader():
    def __init__(self, args, img_loss_list, training_type=TrainingType.ACTIVE_LEARNING) -> None:
        self.args = args
        self.img_loss_list = img_loss_list

        self.training_type = training_type

        params = get_params(args, training_type)
        self.image_size = params.image_size
        self.batch_size = params.batch_size

    def get_loader(self):
        if self.training_type == TrainingType.FINETUNING:
            data_size = len(self.img_loss_list)
            new_data_size = int(self.args.al_finetune_data_ratio * data_size)
            self.img_loss_list = self.img_loss_list[:new_data_size]

        transforms = TransformsSimCLR(self.image_size)
        dataset = FinetuneLoader(self.args, self.img_loss_list, transforms)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        print(f"The size of the dataset is {len(dataset)} and the number of batches is ", loader.__len__())
        return loader


class FinetuneLoader(torch.utils.data.Dataset):
    def __init__(self, args, pathloss_list, transform) -> None:
        self.args = args
        self.pathloss_list = pathloss_list
        self.transform = transform

    def __len__(self):
        return len(self.pathloss_list)

    def __getitem__(self, idx):
        path = self.pathloss_list[idx].path[0]
        img = Image.open(path)
        # img = Image.fromarray(img)

        return self.transform.__call__(img), path

    
class MakeBatchLoader(torch.utils.data.Dataset):
    def __init__(self, image_size, dir, transform=None):
        self.image_size = image_size
        self.img_path = glob.glob(dir + '/*/*')
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        # img = Image.fromarray(img)

        x1, x2 = self.transform.__call__(img)
        return [x1, x2], self.img_path[idx]
