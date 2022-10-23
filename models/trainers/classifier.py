import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import time
import copy

from datautils.finetune_dataset import Finetune
from models.backbones.resnet import resnet_backbone
from models.heads.logloss_head import LogLossHead
from models.utils.commons import get_params_to_update, set_parameter_requires_grad
from utils.commons import accuracy, load_saved_state, simple_save_model


class Classifier:
    def __init__(self, args, pretrain_level="2") -> None: # this can also be called after the base pretraining to evaluate the performance

        self.args = args
        # self.model = LogLossHead(self.encoder, with_avg_pool=True, in_channels=2048, num_classes=None) #todo: The num_classes parameter is determined by the dataset used for the finetuning
        
        self.model = resnet_backbone(self.args.resnet, pretrained=False)
        state = load_saved_state(self.args, pretrain_level=pretrain_level)
        self.model.load_state_dict(state['model'])

        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)
        self.model = self.model.to(self.args.device)

        self.model = self.model.to(self.args.device)

        set_parameter_requires_grad(self.model, feature_extract=True)
        params_to_update = get_params_to_update(self.model, feature_extract=True)

        self.optimizer = torch.optim.SGD(
                            params_to_update, 
                            self.args.finetune_lr,
                            momentum=self.args.finetune_momentum,
                            weight_decay=self.args.finetune_weight_decay)

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)

    def finetune(self) -> None:
        self.model = self.model.to(self.args.device)

        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

        train_loader, val_loader = Finetune(self.args)

        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.args.finetune_start_epoch, self.args.finetune_epochs):
            print('Epoch {}/{}'.format(epoch, (self.args.finetune_start_epoch, self.args.finetune_epochs) - 1))
            print('-' * 10)

            # train for one epoch
            self.train_single_epoch(train_loader, self.model, self.criterion, self.optimizer)

            # evaluate on validation set
            epoch_acc, best_model_wts = self.validate(val_loader, self.model, self.criterion, best_acc)
            val_acc_history.append(epoch_acc)
            
            scheduler.step()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        simple_save_model(self.args, self.model, 'classifier_{:4f}_acc.pth'.format(best_acc))

        return self.model, val_acc_history


    def train_single_epoch(self, train_loader, model, criterion, optimizer) -> None:
        model.train()

        loss = 0.0
        corrects = 0

        for _, (images, targets) in enumerate(train_loader):
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)
            
            optimizer.zero_grad()

            # compute output
            outputs = model(images)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            # statistics
            loss += loss.item() * images.size(0)
            corrects += torch.sum(preds == targets.data)

        epoch_loss, epoch_acc = self.accuracy(loss, corrects, train_loader)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


    def validate(self, val_loader, model, criterion, best_acc) -> None:    
        model.eval()

        loss = 0.0
        corrects = 0

        with torch.no_grad():
            for _, (images, targets) in enumerate(val_loader):
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                # compute output
                outputs = model(images)
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)

                # statistics
                loss += loss.item() * images.size(0)
                corrects += torch.sum(preds == targets.data)

            epoch_loss, epoch_acc = self.accuracy(loss, corrects, val_loader)
            print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        return epoch_acc, best_model_wts

    def accuracy(self, loss, corrects, loader):
        epoch_loss = loss / len(loader.dataset)
        epoch_acc = corrects.double() / len(loader.dataset)

        return epoch_loss, epoch_acc