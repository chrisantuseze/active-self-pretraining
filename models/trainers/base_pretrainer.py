from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.active_learning.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss
from datautils import dataset_enum, cifar10, imagenet

class BasePretrainer():
    def __init__(self, args) -> None:
        self.args = args
    
    def first_pretrain(self) :
        # initialize ResNet
        encoder = resnet_backbone(self.args.backbone, pretrained=False)
        print("=> creating model '{}'".format(self.args.backbone))

        if self.args.base_dataset == dataset_enum.DatasetType.IMAGENET.value:
            train_loader = imagenet.ImageNet(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        elif self.args.base_dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        else:
             train_loader = get_target_pretrain_ds(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader() 

        return encoder, train_loader


    def second_pretrain(self) -> None:
        if self.args.do_al:
            pretrain_data = load_path_loss(self.args, self.args.pretrain_path_loss_file)
            if pretrain_data is None:
                pretext = PretextTrainer(self.args, self.writer)
                pretrain_data = pretext.do_active_learning()

            loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN).get_loader()
        else:
            loader = get_target_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loader()        

        encoder = resnet_backbone(self.args.backbone, pretrained=False)

        return encoder, loader