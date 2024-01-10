import time
import utils.logger as logging
from optim.optimizer import load_optimizer
from models.utils.commons import get_model_criterion, get_params, AverageMeter, prepare_model
from models.utils.training_type_enum import TrainingType

class SimCLRTrainer():
    def __init__(self, 
        args, writer, encoder, 
        dataloader, pretrain_level="1", 
        training_type=TrainingType.BASE_PRETRAIN, log_step=500) -> None:
    
        self.args = args
        self.writer = writer
        self.log_step = log_step

        self.train_loader = dataloader

        self.model, self.criterion = get_model_criterion(self.args, encoder, training_type)
        
        self.model, params_to_update = prepare_model(self.args, training_type, "1", self.model)

        self.model = self.model.to(self.args.device)

        self.train_params = get_params(self.args, training_type)
        self.optimizer, self.scheduler = load_optimizer(self.args, params=params_to_update, train_params=self.train_params)

    def train_epoch(self, epoch) -> int:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # total_loss, total_num = 0.0, 0
        self.model.train()

        end = time.time()

        for step, (inputs, _) in enumerate(self.train_loader):
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # image = image.to(self.args.device)
            output = self.model(inputs)
            loss = self.criterion(output)

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