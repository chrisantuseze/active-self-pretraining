import copy

import torch
import torch.nn as nn
from utils.commons import simple_save_model

import utils.logger as logging
from models.utils.commons import accuracy

class Trainer:
    def __init__(self, args, writer, model, train_loader, val_loader, train_params) -> None:
        self.args = args
        self.writer = writer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_params = train_params

        self.model = self.model.to(self.args.device)
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_params.lr,
            nesterov=False,
            momentum=args.momentum,
            weight_decay=train_params.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, train_params.epochs, eta_min=1e-5)
        self.criterion = nn.CrossEntropyLoss()

        self.best_model = copy.deepcopy(self.model)
        self.best_acc = 0

    def train(self) -> None:

        val_acc_history = []
        lr = self.train_params.lr

        for epoch in range(self.train_params.epochs):
            logging.info('\nEpoch {}/{} lr: {}'.format(epoch, self.train_params.epochs, lr))
            logging.info('-' * 20)

            train_loss, train_acc = self.train_single_epoch()
            val_loss, val_acc = self.validate()
            val_acc_history.append(str(val_acc))

            # Decay Learning Rate
            if self.scheduler:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()

            self.writer.add_scalar(f"{self.train_params.name}/train_loss", train_loss, epoch)
            self.writer.add_scalar(f"{self.train_params.name}/val_loss", val_loss, epoch)

        logging.info('Best val accuracy: {:3f}'.format(self.best_acc))

        simple_save_model(self.args, self.best_model, f'{self.train_params.name}.pth')

        # additional_ext = get_accuracy_file_ext(self.args)
        # save_accuracy_to_file(
        #     self.args, accuracies=val_acc_history, best_accuracy=self.best_acc, 
        #     filename=f"classifier_{get_dataset_enum(self.args.lc_dataset)}_batch_{self.args.lc_epochs}{additional_ext}.txt")

    def train_single_epoch(self):
        self.model.train()

        total_loss, corrects = 0.0, 0
        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.args.log_step == 0:
                logging.info(f"Train Step [{step}/{len(self.train_loader)}]\t Loss: {loss.item()}")

            # statistics
            total_loss += loss.item() * images.size(0)
            corrects += torch.sum(preds == targets.data)

        epoch_loss, epoch_acc = accuracy(total_loss, corrects, self.train_loader)
        epoch_acc = epoch_acc * 100.0
        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc
    
    def validate(self):    
        self.model.eval()

        total_loss, corrects = 0.0, 0
        with torch.no_grad():
            for step, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                # compute output
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                _, preds = torch.max(outputs, 1)

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(self.val_loader)}]\t Loss: {loss.item()}")

                # statistics
                total_loss += loss.item() * images.size(0)
                corrects += torch.sum(preds == targets.data)

            epoch_loss, epoch_acc = accuracy(total_loss, corrects, self.val_loader)
            epoch_acc = epoch_acc * 100.0

            # deep copy the model
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model = copy.deepcopy(self.model)

            logging.info('Val Loss: {:.4f} Acc@1: {:.3f} Best Acc@1 so far: {:.3f}'.format(epoch_loss, epoch_acc, self.best_acc))

        return epoch_loss, epoch_acc