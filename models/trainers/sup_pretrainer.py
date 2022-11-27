import torch.nn as nn
from models.trainers.base_pretrainer import BasePretrainer
from models.utils.commons import get_params
from optim.optimizer import load_optimizer
import utils.logger as logging
from models.utils.training_type_enum import TrainingType
from utils.commons import save_state

class SupPretrainer(BasePretrainer):
    def __init__(self, args, writer) -> None:
        self.args = args
        self.writer = writer

    def train_epoch(self, model, train_loader, criterion, optimizer, train_params) -> int:
        total_loss, total_num = 0, 0
        model.train()

        for step, (image, _) in enumerate(train_loader):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

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
        encoder, train_loader = super().first_pretrain()
        
        self.base_pretrain(encoder, train_loader, self.args.base_epochs, trainingType=TrainingType.BASE_PRETRAIN, optimizer_type=self.args.base_optimizer)


    def second_pretrain(self) -> None:

        # This is a technical debt, but for now supervised learning can only be used in the first pretraining layer. But it can be used for the second now
        encoder, loader = super().second_pretrain()

        self.base_pretrain(encoder, loader, self.args.target_epochs, trainingType=TrainingType.TARGET_PRETRAIN, optimizer_type=self.args.target_optimizer)