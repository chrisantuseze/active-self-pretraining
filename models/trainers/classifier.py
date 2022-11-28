import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import time
import copy
import utils.logger as logging
from datautils.dataset_enum import DatasetType, get_dataset_enum

from datautils.finetune_dataset import Finetune
from models.backbones.resnet import resnet_backbone
from models.heads.logloss_head import LogLossHead
from optim.optimizer import load_optimizer
from models.utils.commons import accuracy, get_ds_num_classes, get_model_criterion, get_params, get_params_to_update, set_parameter_requires_grad
from models.utils.training_type_enum import TrainingType
from models.utils.early_stopping import EarlyStopping
from utils.commons import load_saved_state, save_accuracy_to_file, simple_save_model, simple_load_model


class Classifier:
    def __init__(self, args, writer, pretrain_level="2") -> None: # this can also be called after the base pretraining to evaluate the performance

        self.args = args
        self.writer = writer
        
        self.model = resnet_backbone(self.args.resnet, pretrained=False)

        # if pretrain_level == "AL":
        #     state = simple_load_model(self.args, path=f'proxy_{self.args.al_batches-2}.pth')
        # else:
        #     state = load_saved_state(self.args, pretrain_level=pretrain_level)
            
        state = load_saved_state(self.args, pretrain_level=pretrain_level)
        self.model.load_state_dict(state['model'], strict=False)

        num_classes, self.dir = get_ds_num_classes(self.args.finetune_dataset)

        set_parameter_requires_grad(self.model, feature_extract=True)
        self.model, self.criterion = get_model_criterion(self.args, self.model, TrainingType.FINETUNING, num_classes=num_classes)
        self.model = self.model.to(self.args.device)

        params_to_update = get_params_to_update(self.model, feature_extract=True)

        train_params = get_params(self.args, TrainingType.FINETUNING)
        self.optimizer, self.scheduler = load_optimizer(self.args, params_to_update, state, train_params)

        self.best_model = copy.deepcopy(self.model)
        self.best_acc = 0

    def finetune(self, pretrain_data=None) -> None:
        train_loader, val_loader = Finetune(
            self.args, dir=self.dir, 
            training_type=TrainingType.FINETUNING).get_loader(pretrain_data=pretrain_data)

        since = time.time()

        val_acc_history = []

        early_stopping = EarlyStopping(tolerance=5, min_delta=20)

        for epoch in range(self.args.finetune_epochs):

            logging.info('\nEpoch {}/{} lr: '.format(epoch, self.args.finetune_epochs, self.scheduler.get_last_lr()))
            logging.info('-' * 10)

            # train for one epoch
            train_loss, train_acc = self.train_single_epoch(train_loader)

            # evaluate on validation set
            val_loss, val_acc = self.validate(val_loader)
            val_acc_history.append(str(val_acc))

            # Decay Learning Rate
            if not self.scheduler:
                self.scheduler.step()

            # early stopping
            early_stopping(train_loss, val_loss)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val accuracy: {:3f}'.format(self.best_acc * 100))

        simple_save_model(self.args, self.best_model, 'classifier_{:4f}_acc.pth'.format(self.best_acc))
        save_accuracy_to_file(
            self.args, accuracies=val_acc_history, best_accuracy=self.best_acc, 
            filename=f"classifier_{get_dataset_enum(self.args.finetune_dataset)}_batch_{self.args.finetune_epochs}.txt")

        return self.model, val_acc_history

    def train_single_epoch(self, train_loader):
        self.model.train()

        loss = 0.0
        corrects = 0
        for step, (images, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()

            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # compute output
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            self.optimizer.step()

            if step % self.args.log_step == 0:
                logging.info(f"Train Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            # statistics
            loss += loss.item() * images.size(0)
            corrects += torch.sum(preds == targets.data)

        epoch_loss, epoch_acc = accuracy(loss, corrects, train_loader)
        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc


    def validate(self, val_loader):    
        self.model.eval()

        loss = 0.0
        corrects = 0
        with torch.no_grad():
            for step, (images, targets) in enumerate(val_loader):
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                # compute output
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(val_loader)}]\t Loss: {loss.item()}")

                # statistics
                loss += loss.item() * images.size(0)
                corrects += torch.sum(preds == targets.data)

            epoch_loss, epoch_acc = accuracy(loss, corrects, val_loader)
            logging.info('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > self.best_acc:
                print(f'Saving.. prev best acc = {self.best_acc}, new best acc = {epoch_acc}')
                self.best_acc = epoch_acc
                self.best_model = copy.deepcopy(self.model)

        return epoch_loss, epoch_acc