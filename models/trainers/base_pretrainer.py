import glob
from datautils.path_loss import PathLoss
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.backbones.resnet import resnet_backbone
from models.utils.training_type_enum import TrainingType
from datautils import dataset_enum, imagenet

class BasePretrainer():
    def __init__(self, args) -> None:
        self.args = args
    
    def first_pretrain(self) :
        # initialize ResNet
        encoder = resnet_backbone(self.args.backbone, pretrained=False)
        print("=> creating model '{}'".format(self.args.backbone))

        if self.args.base_dataset == dataset_enum.DatasetType.IMAGENET.value:
            train_loader = imagenet.ImageNet(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        else:
            img_path = glob.glob(f'{self.args.dataset_dir}/{self.args.base_dataset}/*')            
            pretrain_data = [PathLoss(path=sample, loss=0) for sample in img_path]

            train_loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        return encoder, train_loader


    def second_pretrain(self) -> None:
        None