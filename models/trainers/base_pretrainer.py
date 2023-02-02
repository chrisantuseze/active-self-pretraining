from datautils.path_loss import PathLoss
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import MakeBatchDataset, PretextDataLoader
from models.active_learning.pretext_trainer import PretextTrainer
from models.backbones.resnet import resnet_backbone
from models.self_sup.swav.transformation.swav_transformation import TransformsSwAV
from models.utils.ssl_method_enum import SSL_Method
from models.utils.training_type_enum import TrainingType
from models.utils.transformations import Transforms
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
            #  train_loader = get_target_pretrain_ds(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()
            # 

            transforms = Transforms(80)
            dataset = MakeBatchDataset(
                                    self.args,
                                    f'{self.args.dataset_dir}/{self.args.base_dataset}', with_train=False, 
                                    is_train=False, generated=True, transform=transforms) 
            
            pretrain_data = [PathLoss(path=sample, loss=0) for sample in dataset]

            print(pretrain_data[0])

            train_loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN).get_loader()

        return encoder, train_loader


    def second_pretrain(self) -> None:
        None