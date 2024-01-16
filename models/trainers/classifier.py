import torch
import time
import copy
import utils.logger as logging
from datautils.dataset_enum import get_dataset_info

from datautils.lc_dataset import LCDataset
from models.backbones.resnet import resnet_backbone
from optim.optimizer import load_optimizer
from models.utils.commons import accuracy, get_model_criterion, get_params, get_params_to_update, set_parameter_requires_grad
from models.utils.training_type_enum import TrainingType
from utils.commons import get_accuracy_file_ext, load_chkpts, load_saved_state, save_accuracy_to_file, simple_save_model, simple_load_model


class Classifier:
    def __init__(self, args, pretrain_level="2") -> None: # this can also be called after the base pretraining to evaluate the performance
        self.args = args
        self.model = resnet_backbone(self.args.backbone, pretrained=False)

        if pretrain_level == "1":
            state = load_saved_state(args, dataset=get_dataset_info(args.base_dataset)[1], pretrain_level=pretrain_level)

        elif pretrain_level == "AL":
            logging.info("Using pretext task weights")
            state = simple_load_model(self.args, path='finetuner.pth')
        else:
            state = load_saved_state(args, dataset=get_dataset_info(args.target_dataset)[1], pretrain_level=pretrain_level)

        self.model.load_state_dict(state['model'], strict=False)
        num_classes, self.dataset, self.dir = get_dataset_info(self.args.lc_dataset)

        set_parameter_requires_grad(self.model, feature_extract=True)
        self.model, self.criterion = get_model_criterion(self.args, self.model, TrainingType.LINEAR_CLASSIFIER, num_classes=num_classes)
        self.model = self.model.to(self.args.device)

        params_to_update = get_params_to_update(self.model, feature_extract=True)
        train_params = get_params(self.args, TrainingType.LINEAR_CLASSIFIER)
        self.optimizer, self.scheduler = load_optimizer(self.args, params_to_update, train_params)

        self.best_model = copy.deepcopy(self.model)
        self.best_acc = 0

    def train_and_eval(self) -> None:
        train_loader, val_loader = LCDataset(self.args, dir=self.dir, training_type=TrainingType.LINEAR_CLASSIFIER).get_loader()

        since = time.time()
        val_acc_history = []
        logging.info(f"Performing linear eval on {self.dataset}")

        for epoch in range(self.args.lc_epochs):
            lr = 0
            if self.scheduler:
                lr = self.scheduler.get_last_lr()

            logging.info('\nEpoch {}/{} lr: '.format(epoch, self.args.lc_epochs, lr))
            logging.info('-' * 10)

            _, _ = self.train_single_epoch(train_loader)
            _, val_acc = self.validate(val_loader)
            val_acc_history.append(str(val_acc))

            # Decay Learning Rate
            if self.scheduler:
                self.scheduler.step()

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val accuracy: {:3f}'.format(self.best_acc))

        simple_save_model(self.args, self.best_model, 'classifier_{:4f}_acc.pth'.format(self.best_acc))

        additional_ext = get_accuracy_file_ext(self.args)
        save_accuracy_to_file(
            self.args, accuracies=val_acc_history, best_accuracy=self.best_acc, 
            filename=f"classifier_{self.dataset}_batch_{self.args.lc_epochs}{additional_ext}.txt")

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
                images, targets = images.to(self.args.device), targets.to(self.args.device)

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