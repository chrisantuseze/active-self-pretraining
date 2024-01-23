from datautils.dataset_enum import get_dataset_info
from datautils.target_dataset import get_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.trainers.swav import SwAVTrainer
import utils.logger as logging
from models.utils.training_type_enum import TrainingType
from utils.commons import load_path_loss, save_state


class SelfSupPretrainer:
    def __init__(self, 
        args, 
        writer) -> None:

        self.args = args
        self.writer = writer

    def base_pretrain(self, train_loader, epochs, batch, trainingType) -> None:        
        if trainingType == TrainingType.SOURCE_PRETRAIN:
            pretrain_level, plevel = "1", "1"
            dataset_type = get_dataset_info(self.args.source_dataset)[1]
        else:
            pretrain_level, plevel = "2", "2"
            if trainingType == TrainingType.TARGET_AL:
                pretrain_level, plevel = f"2_{batch-1}", f"2_{batch}"
            dataset_type = get_dataset_info(self.args.target_dataset)[1]

        logging.info(f"{trainingType.value} pretraining in progress, please wait...")

        log_step = self.args.log_step
        trainer = SwAVTrainer(
            self.args, train_loader, 
            pretrain_level=pretrain_level,
            training_type=trainingType, 
            log_step=log_step
        )

        model = trainer.model
        self.args.current_epoch = 0
        for epoch in range(epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, epochs))
            logging.info('-' * 20)

            epoch_loss = trainer.train_epoch(epoch)

            # Decay Learning Rate
            if epoch > 1 and epoch % epochs//2 == 0:
                save_state(self.args, model, dataset_type, pretrain_level)

            self.args.current_epoch += 1

        save_state(self.args, model, dataset_type, pretrain_level=plevel)


    def first_pretrain(self) -> None:
        loader = self.get_loader(do_al=False, training_type=TrainingType.SOURCE_PRETRAIN)
        self.base_pretrain(loader, self.args.source_epochs, batch=0, trainingType=TrainingType.SOURCE_PRETRAIN)

    def second_pretrain(self) -> None:
        distilled_ds = load_path_loss(self.args, self.args.pretrain_path_loss_file)
        loader = self.get_loader(self.args.do_al, distilled_ds=distilled_ds, training_type=TrainingType.TARGET_PRETRAIN)
        self.base_pretrain(loader, self.args.target_epochs, batch=0, trainingType=TrainingType.TARGET_PRETRAIN)

    def get_loader(self, do_al, distilled_ds=None, training_type=None):
        if do_al:
            loader = PretextDataLoader(self.args, distilled_ds, training_type=training_type).get_loader()
        else:
            loader = get_pretrain_ds(self.args, training_type=training_type).get_loader()  

        return loader
