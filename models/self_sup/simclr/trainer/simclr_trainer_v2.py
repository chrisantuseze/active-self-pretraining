import time
from datautils.dataset_enum import DatasetType
import utils.logger as logging
from models.self_sup.simclr.loss.dcl_loss import DCL
from optim.optimizer import load_optimizer
from models.utils.commons import get_model_criterion, get_params, AverageMeter, get_params_to_update, prepare_model
from models.utils.training_type_enum import TrainingType

class SimCLRTrainerV2():
    def __init__(self, 
        args, writer, encoder, 
        dataloader, pretrain_level="1", 
        training_type=TrainingType.BASE_PRETRAIN, log_step=500) -> None:
    
        self.args = args
        self.writer = writer
        self.log_step = log_step

        self.train_loader = dataloader

        if self.args.base_dataset == DatasetType.IMAGENET or self.args.target_dataset == DatasetType.IMAGENET:
            self.args.temperature = 0.1

        elif self.args.base_dataset == DatasetType.CIFAR10 or self.args.target_dataset == DatasetType.CIFAR10:
            self.args.temperature = 0.07

        self.model, self.criterion = get_model_criterion(self.args, encoder, training_type)
        
        self.model, params_to_update = prepare_model(self.args, training_type, self.model)

        self.model = self.model.to(self.args.device)

        self.train_params = get_params(self.args, training_type)
        self.optimizer, self.scheduler = load_optimizer(self.args, params=params_to_update, train_params=self.train_params)

    def train_epoch(self, epoch) -> int:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()
        end = time.time()

        for step, (inputs, _) in enumerate(self.train_loader):
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            inputs = inputs.to(self.args.device)

            # Forward pass to get output/logits
            _, output1 = self.model(inputs)
            _, output2 = self.model(inputs)

            # Calculate Loss: softmax --> cross entropy loss
            loss = self.criterion(output1, output2) + self.criterion(output2, output1)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            self.optimizer.step()

            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if step % self.log_step == 0:
                logging.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        step,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                )

            self.args.global_step += 1

        return losses.avg