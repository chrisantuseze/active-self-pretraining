from models.self_sup.simclr.modules.optimizer import load_optimizer
from models.utils.commons import get_model_criterion, get_params
from models.utils.training_type_enum import TrainingType
from utils.commons import load_saved_state

class SimCLRTrainer():
    def __init__(self, 
    args, writer, encoder, 
    train_loader, rebuild_al_model=False, 
    pretrain_level="1", training_type=TrainingType.BASE_PRETRAIN, 
    log_step=250) -> None:
    
        self.args = args
        self.writer = writer
        self.log_step = log_step

        self.train_loader = train_loader

        state = None
        if training_type == TrainingType.ACTIVE_LEARNING and not rebuild_al_model:
            self.model = encoder

        else:
            self.model, self.criterion = get_model_criterion(self.args, encoder, training_type)

            if training_type != TrainingType.BASE_PRETRAIN:
                state = load_saved_state(self.args, pretrain_level=pretrain_level)
                self.model.load_state_dict(state['model'], strict=False)

            self.model = self.model.to(self.args.device)

        train_params = get_params(self.args, training_type)
        self.optimizer, self.scheduler = load_optimizer(self.args, self.model.parameters(), state, train_params)

    def train_epoch(self) -> int:
        loss_epoch = 0
        self.model.train()

        for step, (images, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            images[0] = images[0].to(self.args.device)
            images[1] = images[1].to(self.args.device)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = self.model(images[0], images[1])
            loss = self.criterion(z_i, z_j)

            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            if step % self.log_step == 0:
                print(f"Step [{step}/{len(self.train_loader)}]\t Loss: {loss}")

            self.writer.add_scalar("Loss/train_epoch", loss, self.args.global_step)
            self.args.global_step += 1

            loss_epoch += loss
        return loss_epoch