import torch.nn as nn
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.active_learning.pretext_trainer import PretextTrainer
from models.utils.commons import get_params
from optim.optimizer import load_optimizer
import utils.logger as logging
from models.backbones.resnet import resnet_backbone
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss, load_saved_state, save_state
from datautils import dataset_enum, cifar10, imagenet

class SupervisedPretrainer():
    def __init__(self, args, writer) -> None:
        self.args = args
        self.writer = writer

    def train_epoch(self, model, train_loader, criterion, optimizer, train_params) -> int:
        model.train()

        for step, (image, _) in enumerate(train_loader):
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            image = image.to(self.args.device)
            output = model(image)
            loss = criterion(output)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            total_num += train_params.batch_size
            total_loss += loss.item() * train_params.batch_size

            if step % self.args.log_step == 0:
                logging.info(f"Step [{step}/{len(train_loader)}]\t Loss: {total_loss / total_num}")

            self.writer.add_scalar("Loss/train_epoch", loss, self.args.global_step)
            self.args.global_step += 1

        return total_loss / total_num

    def base_pretrain(self, model, train_loader, epochs, trainingType, optimizer_type) -> None:
        pretrain_level = "1" if trainingType == TrainingType.BASE_PRETRAIN else "2"        
        logging.info(f"{trainingType.value} pretraining in progress, please wait...")

        train_params = get_params(self.args, trainingType)
        optimizer, scheduler = load_optimizer(self.args, model.parameters(), None, train_params)
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        for epoch in range(self.args.start_epoch, epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, (epochs - self.args.start_epoch)))
            logging.info('-' * 20)

            epoch_loss = self.train_epoch(model, train_loader, criterion, optimizer, train_params)

            # Decay Learning Rate
            scheduler.step()

            if epoch > 0 and epoch % 20 == 0:
                save_state(self.args, model, optimizer, pretrain_level, optimizer_type)

            logging.info(f"Epoch Loss: {epoch_loss}\t lr: {scheduler.get_last_lr()}")
            logging.info('-' * 20)

            self.args.current_epoch += 1

        save_state(self.args, model, optimizer, pretrain_level, optimizer_type)


    def first_pretrain(self) -> None:
        # initialize ResNet
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        print("=> creating model '{}'".format(self.args.resnet))

        if self.args.dataset == dataset_enum.DatasetType.IMAGENET.value or self.args.dataset == dataset_enum.DatasetType.IMAGENET_LITE.value:
            train_loader = imagenet.ImageNet(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        elif self.args.dataset == dataset_enum.DatasetType.CIFAR10.value:
            train_loader = cifar10.CIFAR10(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()

        else:
             train_loader = get_target_pretrain_ds(self.args, training_type=TrainingType.BASE_PRETRAIN).get_loader()  

        self.base_pretrain(encoder, train_loader, self.args.base_epochs, trainingType=TrainingType.BASE_PRETRAIN, optimizer_type=self.args.base_optimizer)


    def second_pretrain(self) -> None:
        if self.args.do_al:
            pretrain_data = load_path_loss(self.args, self.args.pretrain_path_loss_file)
            if pretrain_data is None:
                pretext = PretextTrainer(self.args, self.writer)
                pretrain_data = pretext.do_active_learning()

            loader = PretextDataLoader(self.args, pretrain_data, training_type=TrainingType.TARGET_PRETRAIN).get_loader()
        else:
            loader = get_target_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loader()        

        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        self.base_pretrain(encoder, loader, self.args.target_epochs, trainingType=TrainingType.TARGET_PRETRAIN, optimizer_type=self.args.target_optimizer)