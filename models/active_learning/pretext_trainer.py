import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from datautils.dataset_enum import get_dataset_enum
from optim.optimizer import load_optimizer
import utils.logger as logging
from typing import List
import copy
import random

from datautils.path_loss import PathLoss
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import Loader, MakeBatchLoader, PretextDataLoader
from models.backbones.resnet import resnet_backbone

from models.utils.commons import get_ds_num_classes, get_feature_dimensions_backbone, get_model_criterion, get_params
from models.utils.training_type_enum import TrainingType
from models.active_learning.al_method_enum import AL_Method, get_al_method_enum
from utils.commons import load_path_loss, load_saved_state, save_accuracy_to_file, save_path_loss, simple_load_model, simple_save_model

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

    def eval_main_task(self, model, criterion, batch, test_loader):
        model.eval()
        correct, total = 0, 0
        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_num += 100
                total_loss += loss.item() * 100

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(test_loader)}]\t Loss: {total_loss / total_num}\t Acc: {100.*correct/total}")


        # Save checkpoint.
        acc = 100. * correct / total
        self.val_acc_history.append(str(acc))
        if acc > self.best_proxy_acc:
            print(f'Saving.. Prev acc = {self.best_proxy_acc}, new acc = {acc}')
            simple_save_model(self.args, model, f'proxy_{batch}.pth')
            self.best_proxy_acc = acc
            self.best_batch = batch

    def train_main_task(self, model, criterion, optimizer, train_params, train_loader):
        model.train()
        total_loss, total_num = 0.0, 0
        for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()

                total_num += train_params.batch_size
                total_loss += loss.item() * train_params.batch_size

                if step % self.args.log_step == 0:
                    logging.info(f"Train Step [{step}/{len(train_loader)}]\t Loss: {total_loss / total_num}")

    def main_task(self, samples, model, batch, rebuild_al_model=False):
        train_loader = PretextDataLoader(self.args, samples, is_val=False, batch_size=self.args.al_finetune_batch_size).get_loader()
        test_loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=100).get_loader()

        state = None
        _, criterion = get_model_criterion(self.args, model)
        if rebuild_al_model:
            state = simple_load_model(self.args, path='finetuner.pth')
            model.load_state_dict(state['model'], strict=False)
        
        model = model.to(self.args.device)
        train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)
        optimizer, scheduler = load_optimizer(self.args, model.parameters(), state, train_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        self.val_acc_history = []

        for epoch in range(self.args.al_epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.args.al_epochs))
            logging.info('-' * 20)

            self.train_main_task(model, criterion, optimizer, train_params, train_loader)
            self.eval_main_task(model, criterion, batch, test_loader)

            # Decay Learning Rate
            scheduler.step()

        return model

    def batch_sampler(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=1).get_loader()

        logging.info(f"Generating the top1 scores using {get_al_method_enum(self.args.al_method)}")
        _preds = []

        model.eval()
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = model(inputs)

                _preds.append(self.get_predictions(outputs))

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]")

        preds = torch.cat(_preds).numpy()
       
        return self.get_new_samples(preds, samples)

    def get_predictions(self, outputs):
        dist1 = F.softmax(outputs, dim=1)
        preds = dist1.detach().cpu()

        return preds

    def get_new_samples(self, preds, samples) -> List[PathLoss]:
        if self.args.al_method == AL_Method.LEAST_CONFIDENCE.value:
            probs = preds.max(axis=1)
            indices = probs.argsort(axis=0)

        elif self.args.al_method == AL_Method.ENTROPY.value:
            entropy = (np.log(preds) * preds).sum(axis=1) * -1.
            indices = entropy.argsort(axis=0)[::-1]

        elif self.args.al_method == AL_Method.BOTH.value:
            probs = preds.max(axis=1)
            indices1 = probs.argsort(axis=0)

            entropy = (np.log(preds) * preds).sum(axis=1) * -1.
            indices2 = entropy.argsort(axis=0)[::-1]

            indices = np.concatenate((indices1, indices2)) 
            random.shuffle(indices)
            indices = indices[: (len(indices)//2)]

        else:
            raise ValueError(f"'{self.args.al_method}' method doesn't exist")

        new_samples = []
        for item in indices:
            new_samples.append(samples[item]) # Map back to original indices

        return new_samples[:self.args.al_trainer_sample_size]

    def make_batches(self, model):
        loader = get_target_pretrain_ds(self.args, training_type=TrainingType.ACTIVE_LEARNING, is_train=False, batch_size=1).get_loader()

        model, criterion = get_model_criterion(self.args, model, num_classes=4)
        state = simple_load_model(self.args, path='finetuner.pth')
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        pathloss = []

        logging.info("About to begin eval to make batches")
        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(loader):
                inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
                outputs = model(inputs)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                outputs3 = model(inputs3)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(outputs1, targets1)
                loss3 = criterion(outputs2, targets2)
                loss4 = criterion(outputs3, targets3)
                loss = (loss1+loss2+loss3+loss4)/4.
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = loss.item()
                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(loader)}]\t Loss: {loss}")

                pathloss.append(PathLoss(path, loss))
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss, reverse=True)
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        return sorted_samples

    def eval_finetuner(self, model, criterion, test_loader):
        model.eval()
        total_loss, total_num = 0.0, 0
        total, correct = 0, 0

        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(test_loader):
                inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
                outputs = model(inputs)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                outputs3 = model(inputs3)

                loss1 = criterion(outputs, targets)
                loss2 = criterion(outputs1, targets1)
                loss3 = criterion(outputs2, targets2)
                loss4 = criterion(outputs3, targets3)
                loss = (loss1+loss2+loss3+loss4)/4.

                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                total_num += 100
                total_loss += loss.item() * 100

                if step % self.args.log_step == 0:
                    logging.info(f"Eval Step [{step}/{len(test_loader)}]\t Loss: {total_loss / total_num}\t Acc: {100.*correct/total}")

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_trainer_acc:
            print(f'Saving.. prev best acc = {self.best_trainer_acc}, new best acc = {acc}')
            self.best_model = copy.deepcopy(model)
            
            self.best_trainer_acc = acc

    def train_finetuner(self, model, criterion, optimizer, train_loader):
        model.train()
        total_loss, total_num = 0, 0
        total, correct = 0, 0

        for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(train_loader):
            inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
            
            optimizer.zero_grad()
            outputs, outputs1, outputs2, outputs3 = model(inputs), model(inputs1), model(inputs2), model(inputs3)

            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * self.args.al_batch_size
            total_num += self.args.al_batch_size

            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0)*4

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
            correct += predicted3.eq(targets3).sum().item()

            if step % self.args.log_step == 0:
                logging.info(f"Train Step [{step}/{len(train_loader)}]\t Loss: {total_loss / total_num}")

    def finetuner(self, model):
        train_loader = get_target_pretrain_ds(
            self.args, training_type=TrainingType.ACTIVE_LEARNING, 
            is_train=True, batch_size=self.args.al_batch_size).get_loader()

        test_loader = get_target_pretrain_ds(
            self.args, training_type=TrainingType.ACTIVE_LEARNING, 
            is_train=False, batch_size=100).get_loader()

        model, criterion = get_model_criterion(self.args, model, num_classes=4)
        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)
        optimizer, scheduler = load_optimizer(self.args, model.parameters(), state, train_params)

        for epoch in range(self.args.al_finetune_trainer_epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.args.al_finetune_trainer_epochs))
            logging.info('-' * 20)

            self.train_finetuner(model, criterion, optimizer, train_loader)
            self.eval_finetuner(model, criterion, test_loader)

            scheduler.step()

        simple_save_model(self.args, self.best_model, 'finetuner.pth')


    def do_active_learning(self) -> List[PathLoss]:
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        
        main_task_model = encoder
        # main_task_model.linear = nn.Linear(self.n_features, self.num_classes)

        state = simple_load_model(self.args, path='finetuner.pth')
        if not state:
            self.finetuner(encoder)

        path_loss = load_path_loss(self.args, self.args.al_path_loss_file)
        if path_loss is None:
            path_loss = self.make_batches(encoder)

        pretraining_sample_pool = []
        rebuild_al_model = True

        sample_per_batch = len(path_loss)//self.args.al_batches
        for batch in range(0, self.args.al_batches):
            sample6400 = path_loss[batch * sample_per_batch : (batch + 1) * sample_per_batch]

            if batch > 0:
                logging.info(f'>> Getting best checkpoint for batch {batch + 1}')

                state = simple_load_model(self.args, f'proxy_{self.best_batch}.pth')
                main_task_model.load_state_dict(state['model'], strict=False)

                # sampling
                samplek = self.batch_sampler(main_task_model, sample6400)
            else:
                # first iteration: sample k at even intervals
                samplek = sample6400[:self.args.al_trainer_sample_size]

            pretraining_sample_pool.extend(samplek)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                main_task_model = self.main_task(
                    pretraining_sample_pool, 
                    main_task_model, batch, rebuild_al_model=rebuild_al_model)

                rebuild_al_model=False

        logging.info('Best main task val accuracy: {:3f} for {}'.format(self.best_proxy_acc, get_al_method_enum(self.args.al_method)))
        save_accuracy_to_file(
                self.args, accuracies=self.val_acc_history, best_accuracy=self.best_proxy_acc, 
                filename=f"main_task_{get_dataset_enum(self.args.target_dataset)}_{get_al_method_enum(self.args.al_method)}_batch_{self.args.al_epochs}.txt")
        save_path_loss(self.args, self.args.pretrain_path_loss_file, pretraining_sample_pool)
        return pretraining_sample_pool