import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import time
import copy
import utils.logger as logging
from datautils.dataset_enum import DatasetType

from datautils.finetune_dataset import Finetune
from models.backbones.resnet import resnet_backbone
from models.heads.logloss_head import LogLossHead
from optim.optimizer import load_optimizer
from models.utils.commons import accuracy, get_params, get_params_to_update, set_parameter_requires_grad
from models.utils.training_type_enum import TrainingType
from models.utils.early_stopping import EarlyStopping
from utils.commons import load_saved_state, simple_save_model, simple_load_model


class Classifier:
    def __init__(self, args, writer, pretrain_level="2") -> None: # this can also be called after the base pretraining to evaluate the performance

        self.args = args
        self.writer = writer
        # self.model = LogLossHead(self.encoder, with_avg_pool=True, in_channels=2048, num_classes=None) #todo: The num_classes parameter is determined by the dataset used for the finetuning
        
        self.model = resnet_backbone(self.args.resnet, pretrained=False)

        if pretrain_level == "AL":
            state = simple_load_model(self.args, path=f'proxy_{self.args.al_batches-1}.pth')
        else:
            state = load_saved_state(self.args, pretrain_level=pretrain_level)
            
        self.model.load_state_dict(state['model'], strict=False)

        n_features = self.model.fc.in_features
        
        if args.finetune_dataset == DatasetType.CLIPART.value:
            num_classes = 345
            self.dir = "/clipart"
            
        elif args.finetune_dataset == DatasetType.SKETCH.value:
            num_classes = 345
            self.dir = "/sketch"

        elif args.finetune_dataset == DatasetType.QUICKDRAW.value:
            num_classes = 345
            self.dir = "/quickdraw"

        elif args.finetune_dataset == DatasetType.UCMERCED.value:
            num_classes = 21
            self.dir = "/ucmerced/images"

        elif args.finetune_dataset == DatasetType.IMAGENET.value:
            num_classes = 200
            self.dir = "/imagenet"

        elif args.finetune_dataset == DatasetType.CIFAR10.value:
            num_classes = 10
            self.dir = "/cifar10"
        
        else: 
            NotImplementedError

        set_parameter_requires_grad(self.model, feature_extract=True)
        self.model.fc = nn.Linear(n_features, num_classes)
        self.model = self.model.to(self.args.device)

        params_to_update = get_params_to_update(self.model, feature_extract=True)

        train_params = get_params(self.args, TrainingType.FINETUNING)
        self.optimizer, self.scheduler = load_optimizer(self.args, params_to_update, state, train_params)

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)

    def finetune(self) -> None:
        train_loader, val_loader = Finetune(self.args, dir=self.dir, training_type=TrainingType.FINETUNING).get_loader()

        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        early_stopping = EarlyStopping(tolerance=5, min_delta=20)

        for epoch in range(self.args.finetune_epochs):

            logging.info('\nEpoch {}/{} lr: '.format(epoch, self.args.finetune_epochs, self.scheduler.get_last_lr()))
            logging.info('-' * 10)

            # train for one epoch
            train_loss, train_acc = self.train_single_epoch(train_loader, self.model, self.criterion, self.optimizer)

            # evaluate on validation set
            val_loss, val_acc, best_acc, best_model_wts = self.validate(val_loader, self.model, self.criterion, best_acc, best_model_wts)
            val_acc_history.append(val_acc)

            # Decay Learning Rate
            self.scheduler.step()

            # early stopping
            early_stopping(train_loss, val_loss)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:3f}'.format(best_acc * 100))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        simple_save_model(self.args, self.model, 'classifier_{:4f}_acc.pth'.format(best_acc))

        return self.model, val_acc_history

    def train_single_epoch(self, train_loader, model, criterion, optimizer) -> None:
        model.train()

        loss = 0.0
        corrects = 0
        for step, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # compute output
            outputs = model(images)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                logging.info(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            # statistics
            loss += loss.item() * images.size(0)
            corrects += torch.sum(preds == targets.data)

        epoch_loss, epoch_acc = accuracy(loss, corrects, train_loader)
        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc


    def validate(self, val_loader, model, criterion, best_acc, best_model_wts) -> None:    
        model.eval()

        loss = 0.0
        corrects = 0
        with torch.no_grad():
            for step, (images, targets) in enumerate(val_loader):
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                # compute output
                outputs = model(images)
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)

                if step % 500 == 0:
                    logging.info(f"Step [{step}/{len(val_loader)}]\t Loss: {loss.item()}")

                # statistics
                loss += loss.item() * images.size(0)
                corrects += torch.sum(preds == targets.data)

            epoch_loss, epoch_acc = accuracy(loss, corrects, val_loader)
            logging.info('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        return epoch_loss, epoch_acc, best_acc, best_model_wts