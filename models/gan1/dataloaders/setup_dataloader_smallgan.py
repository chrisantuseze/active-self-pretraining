import glob
from models.gan1.dataloaders.ImageListDataset import ImageListDataset
from torchvision import transforms
from torch.utils.data import  DataLoader

def setup_dataloader(dir, h=128, w=128, batch_size=4, num_workers=4):
    '''
    instead of setting up dataloader that read raw image from file, 
    let's use store all images on cpu memmory
    because this is for small dataset
    '''

    with_train = True if dir in ["./datasets/chest_xray"] else False
    
    # if with_train:
    #     if dir in ["./datasets/imagenet", "./datasets/chest_xray"]:
    #         img_path_list = glob.glob(dir + '/train/*/*')
    #     else:
    #         img_path_list = glob.glob(dir + '/train/*/*')
    # else:
    #     img_path_list = glob.glob(dir + '/*/*')

    #for imagenet_gan
    img_path_list = glob.glob(dir + '/*')

        
    assert len(img_path_list) > 0

    transform = transforms.Compose([
        transforms.Resize(min(h, w)),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
    ])

    # img_path_list = img_path_list[:25]
    
    img_path_list = [[path, i] for i, path in enumerate(sorted(img_path_list))]
    dataset = ImageListDataset(img_path_list, transform=transform)

    # dataset = MakeBatchDataset(
    #         self.args,
    #         self.dir, self.with_train, self.is_train, transforms)
    
    return DataLoader(
            [data for data in  dataset], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers)