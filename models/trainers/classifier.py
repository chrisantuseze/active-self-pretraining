import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import time
import copy
import utils.logger as logging
from datautils.dataset_enum import DatasetType, get_dataset_enum

from datautils.lc_dataset import LCDataset
from models.backbones.resnet import resnet_backbone
from models.heads.logloss_head import LogLossHead
from optim.optimizer import load_optimizer
from models.utils.commons import accuracy, get_ds_num_classes, get_model_criterion, get_params, get_params_to_update, set_parameter_requires_grad
from models.utils.training_type_enum import TrainingType
from models.utils.early_stopping import EarlyStopping
from utils.commons import get_accuracy_file_ext, get_office_31_model, load_chkpts, load_saved_state, save_accuracy_to_file, simple_save_model, simple_load_model


class Classifier:
    def __init__(self, args, pretrain_level="2") -> None: # this can also be called after the base pretraining to evaluate the performance

        self.args = args
        
        self.model = resnet_backbone(self.args.backbone, pretrained=False)

        if pretrain_level == "AL":
            logging.info("Using pretext task weights")
            state = simple_load_model(self.args, path='finetuner.pth')

        elif args.training_type == "tacc":
            logging.info(f"Using pretrained office-31 model weights")
            state = get_office_31_model(self.args)

        else:
            logging.info(f"Using pretrained {pretrain_level} model weights")
            state = load_saved_state(self.args, pretrain_level=pretrain_level)
            
        if self.args.target_pretrain or self.args.base_pretrain:
            logging.info("<Target Pretrain>")
            self.model.load_state_dict(state['model'], strict=False)
        else:
            logging.info("<SwAV Weights>")
            self.model = load_chkpts(self.args, "swav_800ep_pretrain.pth.tar", self.model)

        num_classes, self.dir = get_ds_num_classes(self.args.lc_dataset)

        set_parameter_requires_grad(self.model, feature_extract=True)
        self.model, self.criterion = get_model_criterion(self.args, self.model, TrainingType.LINEAR_CLASSIFIER, num_classes=num_classes)
        self.model = self.model.to(self.args.device)

        params_to_update = get_params_to_update(self.model, feature_extract=True)

        train_params = get_params(self.args, TrainingType.LINEAR_CLASSIFIER)
        self.optimizer, self.scheduler = load_optimizer(self.args, params_to_update, state, train_params)

        self.best_model = copy.deepcopy(self.model)
        self.best_acc = 0

    def train_and_eval(self, pretrain_data=None) -> None:
        train_loader, val_loader = LCDataset(
            self.args, dir=self.dir, 
            training_type=TrainingType.LINEAR_CLASSIFIER).get_loader(pretrain_data=pretrain_data)

        since = time.time()

        val_acc_history = []

        early_stopping = EarlyStopping(tolerance=10, min_delta=20)

        logging.info(f"Performing linear eval on {get_dataset_enum(self.args.lc_dataset)}")

        for epoch in range(self.args.lc_epochs):

            lr = 0
            if self.scheduler:
                lr = self.scheduler.get_last_lr()

            logging.info('\nEpoch {}/{} lr: '.format(epoch, self.args.lc_epochs, lr))
            logging.info('-' * 10)

            # train for one epoch
            train_loss, train_acc = self.train_single_epoch(train_loader)

            # evaluate on validation set
            val_loss, val_acc = self.validate(val_loader)
            val_acc_history.append(str(val_acc))

            # Decay Learning Rate
            if self.scheduler:
                self.scheduler.step()

            # early stopping
            early_stopping(train_loss, val_loss)
            # if early_stopping.early_stop:
            #     print("We are at epoch:", epoch)
            #     break

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val accuracy: {:3f}'.format(self.best_acc))

        simple_save_model(self.args, self.best_model, 'classifier_{:4f}_acc.pth'.format(self.best_acc))

        additional_ext = get_accuracy_file_ext(self.args)
        save_accuracy_to_file(
            self.args, accuracies=val_acc_history, best_accuracy=self.best_acc, 
            filename=f"classifier_{get_dataset_enum(self.args.lc_dataset)}_batch_{self.args.lc_epochs}{additional_ext}.txt")

        return self.model, val_acc_history

    def train_single_epoch(self, train_loader):
        self.model.train()

        total_loss, corrects = 0.0, 0
        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.args.device), targets.to(self.args.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            self.optimizer.step()

            if step % self.args.log_step == 0:
                logging.info(f"Train Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            # statistics
            total_loss += loss.item() * images.size(0)
            corrects += torch.sum(preds == targets.data)

        epoch_loss, epoch_acc = accuracy(total_loss, corrects, train_loader)
        epoch_acc = epoch_acc * 100.0
        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc


    def validate(self, val_loader):    
        self.model.eval()

        total_loss, corrects = 0.0, 0
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
                total_loss += loss.item() * images.size(0)
                corrects += torch.sum(preds == targets.data)

            epoch_loss, epoch_acc = accuracy(total_loss, corrects, val_loader)
            epoch_acc = epoch_acc * 100.0

            # deep copy the model
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model = copy.deepcopy(self.model)

            logging.info('Val Loss: {:.4f} Acc@1: {:.3f} Best Acc@1 so far: {:.3f}'.format(epoch_loss, epoch_acc, self.best_acc))

        return epoch_loss, epoch_acc