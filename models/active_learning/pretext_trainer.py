import glob
import os
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
        
        data_dir = os.path.join(args.gen_images_path, get_dataset_enum(args.target_dataset))
        gen_images = glob.glob(f'{data_dir}/*/*')

        self.pretraining_gen_images = []
        # for path in gen_images:
        #     label = path.split('/')[-2]
        #     # logging.info("label:", label)
        #     self.pretraining_gen_images.append(PathLoss(path=path, loss=0, label=label))

    def do_self_learning(self):
        encoder = resnet_backbone(self.args, self.num_classes, pretrained=False)
        
        # state = simple_load_model(self.args, path=f'target_{self.dataset}.pth')
        # if not state:
        state = simple_load_model(self.args, path=f'source_{get_dataset_enum(self.args.source_dataset)}.pth')['model']
        # state['prototypes.weight'] = state['prototypes.weight'][:self.num_classes]
        encoder.load_state_dict(state, strict=False)

        self.train_target(encoder, path_loss_list=[], suffix="-1")
        return self.self_learning(encoder)
        
    
    def train_target(self, model, path_loss_list, suffix=""):
        if len(path_loss_list) > 0:
            train_loader, val_loader = PretextDataLoader(
            self.args, path_loss_list, training_type=TrainingType.TARGET_PRETRAIN, is_val=False).get_loaders()
        else:
            train_loader, val_loader = get_pretrain_ds(self.args, training_type=TrainingType.TARGET_PRETRAIN).get_loaders()         
        
        train_params = get_params(self.args, TrainingType.TARGET_PRETRAIN)
        train_params.name = f'target_{self.dataset}{suffix}'

        print("train_params.name", train_params.name)
        trainer = Trainer(self.args, self.writer, model, train_loader, val_loader, train_params)
        trainer.train()

    def make_batches(self, model, suffix=""):
        loader = get_pretrain_ds(self.args, training_type=TrainingType.ACTIVE_LEARNING, is_train=False, batch_size=1).get_loader()

        criterion = torch.nn.CrossEntropyLoss()
        state = simple_load_model(self.args, path=f'target_{self.dataset}{suffix}.pth')
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        pathloss = []
        logging.info("About to begin eval to make batches")

        model.eval()
        with torch.no_grad():
            for step, (inputs, targets, path) in enumerate(loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss.item()

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]\t Loss: {loss}")

                if isinstance(path, tuple) or isinstance(path, list):
                    path = path[0]

                pathloss.append(PathLoss(path=path, loss=loss))
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss) # sort the losses from low to high
        # save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        return sorted_samples
    
    def label_target(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=1).get_loader()

        logging.info("Generating the top1 scores")
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

        preds = torch.cat(_preds)
       
        return self.get_new_data(preds, samples)
    
    def get_new_data(self, preds, samples, threshold=0.5):
        # Select high confidence samples
        conf = preds.max(dim=1)[0]
        masks = conf > threshold
        
        # Get pseudo_labels for high conf samples
        pseudo_labels = preds[masks]
        logging.info("len(pseudo_labels)", len(pseudo_labels))
        if len(pseudo_labels) == 0:
            return []

        pseudo_labels = torch.argmax(pseudo_labels, dim=1)
        
        # Filter objects based on the boolean mask
        new_data = [obj for obj, mask in zip(samples, masks) if mask]
        new_labels = pseudo_labels

        # Update the 'label' attribute of each object in the list
        for i, path_loss_obj in enumerate(new_data):
            path_loss_obj.label = new_labels[i].item()

        return new_data

    def self_learning(self, encoder):
        path_loss = self.make_batches(encoder, "-1")
        path_loss = path_loss[::-1][:10000] # this does a reverse active learning to pick only the most certain data
        logging.info(f"Size of the original data is {len(path_loss)}")

        pretraining_sample_pool = []
        # logging.info(f"Using a pretrain size of {self.args.al_trainer_sample_size} per AL batch.")

        sample_per_batch = len(path_loss)//self.args.al_batches
        model = encoder

        train_params = self.train_params
        train_params.name = f'target_{self.dataset}'

        for batch in range(self.args.al_batches):
            logging.info(f'Batch {batch}')

            state = simple_load_model(self.args, path=f'{train_params.name}{str(batch-1)}.pth')
            model.load_state_dict(state['model'], strict=False)

            samples = path_loss[batch * sample_per_batch : (batch + 1) * sample_per_batch]

            # sampling
            samplek = self.label_target(model, samples)#[:self.args.al_trainer_sample_size]
            pretraining_sample_pool.extend(samplek)

            logging.info(f"Size of pretraining_sample_pool is {len(pretraining_sample_pool)}")

            # retrain target using labeled target data
            self.train_target(model, pretraining_sample_pool, suffix=str(batch))
            model = encoder