import glob
import torch
import torch.nn.functional as F
import numpy as np
from datautils.dataset_enum import get_dataset_enum
import utils.logger as logging
from typing import List

from datautils.path_loss import PathLoss
from datautils.target_dataset import get_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.trainers.resnet import resnet_backbone

from models.utils.commons import get_ds_num_classes, get_feature_dimensions_backbone, get_model_criterion, get_params
from models.utils.training_type_enum import TrainingType
from models.active_learning.al_method_enum import get_al_method_enum
from utils.commons import save_path_loss, simple_load_model


from models.trainers.trainer import Trainer

class PretextTrainer():
    def __init__(self, args, writer) -> None:
        self.args = args
        self.writer = writer
        self.criterion = None
        self.best_proxy_acc = 0
        self.best_batch = 0
        self.best_trainer_acc = 0
        self.val_acc_history = []
        self.best_model = None

        self.num_classes, self.dir = get_ds_num_classes(self.args.target_dataset)
        self.n_features = get_feature_dimensions_backbone(self.args)
        self.dataset = get_dataset_enum(self.args.target_dataset)

        self.train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)

        gen_folder = f'generated_{get_dataset_enum(self.args.target_dataset)}'
        gen_images = glob.glob(f'{self.args.dataset_dir}/{gen_folder}/*/*')
        # self.pretraining_gen_images = [PathLoss(path=path, loss=0) for path in gen_images]

        self.pretraining_gen_images = []
        for path in gen_images:
            label = path.split('/')[-2]
            logging.info("label:", label)
            self.pretraining_gen_images.append(PathLoss(path=path, loss=0, label=label))

    def do_self_learning(self):
        encoder = resnet_backbone(self.args.backbone, pretrained=False)
        
        state = simple_load_model(self.args, path=f'target_{self.dataset}.pth')
        if not state:
            self.train_target(encoder, path_loss_list=[])

        return self.self_learning(encoder)
    
    def train_target(self, model, path_loss_list):
        path_loss_list = self.pretraining_gen_images + path_loss_list
        
        train_loader, val_loader = PretextDataLoader(
            self.args, path_loss_list, training_type=TrainingType.TARGET_PRETRAIN, is_val=False).get_loaders()

        train_params = get_params(self.args, TrainingType.TARGET_PRETRAIN)
        train_params.name = f'target_{self.dataset}'

        trainer = Trainer(self.args, self.writer, model, train_loader, val_loader, train_params)
        trainer.train()

    def make_batches(self, model):
        loader = get_pretrain_ds(self.args, training_type=TrainingType.ACTIVE_LEARNING, is_train=False, batch_size=1).get_loader()

        model, criterion = get_model_criterion(self.args, model, num_classes=4)

        state = simple_load_model(self.args, path=f'target_{self.dataset}.pth')
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        correct = 0
        total = 0
        pathloss = []

        logging.info("About to begin eval to make batches")

        model.eval()
        with torch.no_grad():
            for step, (inputs, targets, path) in enumerate(loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = loss.item()

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]\t Loss: {loss}")

                if isinstance(path, tuple) or isinstance(path, list):
                    path = path[0]

                pathloss.append(PathLoss(path=path, loss=loss))
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss) # sort the losses from low to high
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        return sorted_samples
    
    def label_target(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=1).get_loader()

        logging.info(f"Generating the top1 scores using {get_al_method_enum(self.args.al_method)}")
        _preds = []

        model = model.to(self.args.device)
        model.eval()
        with torch.no_grad():
            for step, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.args.device)
                outputs = model(inputs)

                pred = F.softmax(outputs, dim=1).detach().cpu()
                _preds.append(pred)

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]")

        preds = torch.cat(_preds)#.numpy()
       
        # return self.get_new_samples(preds, samples)
        return self.get_new_data(preds, 0.7, samples)
    
    def get_new_data(self, preds, threshold, samples):
        # Select high confidence samples
        conf = preds.max(dim=1)[0]
        mask = conf > threshold
        
        # Get pseudo_labels for high conf samples
        pseudo_labels = preds[mask]
        logging.info("pseudo_labels:", pseudo_labels)
        
        # Use pseudo_labels as labels for new unlab data
        new_data = samples[mask]
        new_labels = pseudo_labels

        # Update the 'label' attribute of each object in the list
        for i, path_loss_obj in enumerate(new_data):
            path_loss_obj.label = new_labels[i]

        # Combine labeled data  
        # combined_data = torch.concat([gen_target_data, new_data], dim=0)
        # combined_labels = torch.concat([gen_labels, new_labels], dim=0)

        return new_data


    def get_new_samples(self, preds, samples) -> List[PathLoss]:
        entropy = (np.log(preds) * preds).sum(axis=1) * -1.
        indices = entropy.argsort(axis=0)[::-1]

        new_samples = []
        for item in indices:
            new_samples.append(samples[item]) # Map back to original indices

        return new_samples
    
    def self_learning(self, encoder):
        path_loss = self.make_batches(encoder)
        path_loss = path_loss[::-1][:10000] # this does a reverse active learning to pick only the most certain data
        logging.info(f"Size of the original data is {len(path_loss)}")

        pretraining_sample_pool = []

        logging.info(f"Size of pretraining_sample_pool is {len(pretraining_sample_pool)}")
        logging.info(f"Using a pretrain size of {self.args.al_trainer_sample_size} per AL batch.")

        sample_per_batch = len(path_loss)//self.args.al_batches
        model = encoder

        train_params = self.train_params
        train_params.name = f'target_{self.dataset}'

        for batch in range(self.args.al_batches):
            logging.info(f'Batch {batch}')
            samples = path_loss[batch * sample_per_batch : (batch + 1) * sample_per_batch]

            state = simple_load_model(self.args, path=f'{train_params.name}.pth')
            model.load_state_dict(state['model'], strict=False)

            # sampling
            samplek = self.label_target(model, samples)[:self.args.al_trainer_sample_size]
            model = encoder

            pretraining_sample_pool.extend(samplek)

            logging.info(f"Size of pretraining_sample_pool is {len(pretraining_sample_pool)}")

            # retrain target using labeled target data
            state = simple_load_model(self.args, path=f'{train_params.name}.pth')
            model.load_state_dict(state['model'], strict=False)

            self.train_target(model, pretraining_sample_pool)
            model = encoder