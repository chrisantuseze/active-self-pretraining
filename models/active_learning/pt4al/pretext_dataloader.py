from email import utils
import torch

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
    def __init__(self, args, img_loss_list, finetune=False) -> None:
        self.args = args
        self.img_loss_list = img_loss_list
        self.finetune = finetune
        self.batch_size = args.al_batch_size

    def get_loader(self):
        if self.finetune:
            data_size = len(self.img_loss_list)
            new_data_size = int(self.args.al_finetune_data_ratio * data_size)
            self.img_loss_list = self.img_loss_list[:new_data_size]

        dataset = PretextDataset(self.img_loss_list)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return loader

    

