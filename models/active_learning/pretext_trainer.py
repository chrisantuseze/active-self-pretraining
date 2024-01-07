import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from datautils.dataset_enum import get_dataset_enum
from models.trainers.selfsup_pretrainer import SelfSupPretrainer
from models.utils.ssl_method_enum import SSL_Method
from optim.optimizer import load_optimizer
import utils.logger as logging
from typing import List
import copy
import random

from datautils.path_loss import PathLoss
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.backbones.resnet import resnet_backbone

from models.utils.commons import AverageMeter, get_ds_num_classes, get_feature_dimensions_backbone, get_model_criterion, get_params
from models.utils.training_type_enum import TrainingType
from models.active_learning.al_method_enum import AL_Method, get_al_method_enum
from utils.commons import get_state_for_da, load_chkpts, load_path_loss, load_saved_state, save_path_loss, simple_load_model, simple_save_model

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

    def eval_main_task(self, model, epoch, criterion, batch, test_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()

        model.eval()
        correct, total = 0, 0

        end = time.time()
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                losses.update(loss.item(), inputs[0].size(0))
                
                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()

        epoch_acc = 100. * correct / total
        
        # Save checkpoint.
        self.val_acc_history.append(str(epoch_acc))
        if epoch_acc > self.best_proxy_acc:
            simple_save_model(self.args, model, f'proxy_{batch}.pth')
            self.best_proxy_acc = epoch_acc
            self.best_batch = batch

        logging.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=epoch_acc, acc=self.best_proxy_acc))

    def train_main_task(self, model, epoch, criterion, optimizer, train_params, train_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        end = time.time()

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

            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if step % self.args.log_step == 0:
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
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

    def main_task(self, samples, model, batch, rebuild_al_model=False):
        train_loader = PretextDataLoader(self.args, samples, is_val=False, batch_size=self.args.al_maintask_batch_size).get_loader()
        test_loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=100).get_loader()

        state = None
        criterion = nn.CrossEntropyLoss()
        if rebuild_al_model:
            state = simple_load_model(self.args, path='finetuner.pth')
            model.load_state_dict(state['model'], strict=False)

            model.linear = nn.Linear(self.n_features, self.num_classes) # TODO this is a tech debt to figure out why AL complains when we do model.fc instead of model.linear
        
        model = model.to(self.args.device)
        train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)
        optimizer, scheduler = load_optimizer(self.args, model.parameters(), state, train_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        self.val_acc_history = []

        for epoch in range(self.args.al_epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.args.al_epochs))
            logging.info('-' * 20)

            self.train_main_task(model, epoch, criterion, optimizer, train_params, train_loader)
            self.eval_main_task(model, epoch, criterion, batch, test_loader)

            # Decay Learning Rate
            scheduler.step()

        return model

    def batch_sampler(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        loader = PretextDataLoader(self.args, samples, is_val=True, batch_size=1).get_loader()

        logging.info(f"Generating the top1 scores using {get_al_method_enum(self.args.al_method)}")
        _preds = []

        model = model.to(self.args.device)
        model.eval()
        with torch.no_grad():
            for step, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.args.device)
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

        return new_samples

    def make_batches(self, model, prefix, training_type=TrainingType.ACTIVE_LEARNING):
        loader = get_target_pretrain_ds(self.args, training_type=training_type, is_train=False, batch_size=1).get_loader()

        model, criterion = get_model_criterion(self.args, model, num_classes=4)
        state = simple_load_model(self.args, path=f'{prefix}_finetuner_{self.dataset}.pth')
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

                if isinstance(path, tuple) or isinstance(path, list):
                    path = path[0]

                pathloss.append(PathLoss(path, loss))
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss, reverse=False)#reverse=True)
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        return sorted_samples

    def eval_finetuner(self, model, criterion, test_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()

        model.eval()
        end = time.time()
        total, correct = 0, 0

        total_steps = 0
        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(test_loader):
                inputs, inputs1, targets, targets1 = inputs.to(self.args.device), inputs1.to(self.args.device), targets.to(self.args.device), targets1.to(self.args.device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(self.args.device), inputs3.to(self.args.device), targets2.to(self.args.device), targets3.to(self.args.device)
                
                outputs = model(inputs)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                outputs3 = model(inputs3)

                loss = criterion(outputs, targets)
                loss1 = criterion(outputs1, targets1)
                loss2 = criterion(outputs2, targets2)
                loss3 = criterion(outputs3, targets3)
                loss_avg = (loss + loss1 + loss2 + loss3) / 4.

                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                losses.update(loss_avg.item(), inputs[0].size(0))
                
                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()

                total_steps = step

        # Save checkpoint.
        epoch_acc = 100. * correct / total

        if epoch_acc > self.best_trainer_acc:
            self.best_model = copy.deepcopy(model)
            self.best_trainer_acc = epoch_acc

        avg_loss = losses.sum/total_steps
        logging.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss:.4f}\t"
            "Acc@1 {top1:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=avg_loss, top1=epoch_acc, acc=self.best_trainer_acc))

        return epoch_acc, avg_loss


    def train_finetuner(self, model, epoch, criterion, optimizer, scheduler, train_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        end = time.time()
        total_steps = 0
        for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(train_loader):
            
            # update learning rate
            if self.args.al_optimizer == "SwAV":
                scheduler.step(epoch, step)

            inputs, inputs1 = inputs.to(self.args.device), inputs1.to(self.args.device)
            targets, targets1 = targets.to(self.args.device), targets1.to(self.args.device)
            inputs2, inputs3 = inputs2.to(self.args.device), inputs3.to(self.args.device)
            targets2, targets3 = targets2.to(self.args.device), targets3.to(self.args.device)

            optimizer.zero_grad()
            outputs, outputs1, outputs2, outputs3 = model(inputs), model(inputs1), model(inputs2), model(inputs3)

            loss = criterion(outputs, targets)
            loss1 = criterion(outputs1, targets1)
            loss2 = criterion(outputs2, targets2)
            loss3 = criterion(outputs3, targets3)
            loss_avg = (loss + loss1 + loss2 + loss3) / 4.
            loss_avg.backward()
            optimizer.step()

            losses.update(loss_avg.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if step % self.args.log_step == 0:
                logging.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch, step,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses, lr=optimizer.param_groups[0]["lr"],
                    )
                )

            total_steps = step
        avg_loss = losses.sum/total_steps
            
        logging.info("Train Loss: {:.4f}".format(avg_loss))
        return avg_loss

    def finetuner_new(self, model, prefix, path_list: List[PathLoss], training_type=TrainingType.ACTIVE_LEARNING):
        if path_list is not None:
            path_list = [path.path for path in path_list]

            train_loader, test_loader = get_target_pretrain_ds(self.args, training_type=training_type).get_finetuner_loaders(
                train_batch_size=self.args.al_finetune_batch_size, val_batch_size=100, path_list=path_list
            )

        else:
            train_loader, test_loader = get_target_pretrain_ds(self.args, training_type=training_type).get_finetuner_loaders(
                train_batch_size=self.args.al_finetune_batch_size, val_batch_size=100
            )

        model, criterion = get_model_criterion(self.args, model, num_classes=4)
        model = model.to(self.args.device)

        train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)
        optimizer, scheduler = load_optimizer(self.args, model.parameters(), train_params=train_params, train_loader=train_loader)

        counter = 0
        epochs = self.args.al_finetune_trainer_epochs
        logging.info("Running finetuner")
        for epoch in range(epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, epochs))
            logging.info('-' * 20)

            train_loss = self.train_finetuner(model, epoch, criterion, optimizer, scheduler, train_loader)
            epoch_acc, eval_loss = self.eval_finetuner(model, criterion, test_loader)

            # update learning rate
            if self.args.al_optimizer != "SwAV":
                scheduler.step()

            if epoch_acc <= self.best_trainer_acc:
                counter += 1
            else:
                counter = 0

            if counter > 20:
                logging.info("Early stopped at epoch {}:".format(epoch))
                break

        simple_save_model(self.args, self.best_model, f'{prefix}_finetuner_{self.dataset}.pth')

    def finetuner(self, model, prefix, training_type=TrainingType.ACTIVE_LEARNING):
        train_loader, test_loader = get_target_pretrain_ds(
            self.args, training_type=training_type).get_finetuner_loaders(
                train_batch_size=self.args.al_finetune_batch_size,
                val_batch_size=100
            )

        model, criterion = get_model_criterion(self.args, model, num_classes=4)

        state = None
        if self.args.al_pretext_from_pretrain:
            if self.args.backbone == "resnet50" and self.args.method is SSL_Method.SWAV.value:
                model = load_chkpts(self.args, "swav_800ep_pretrain.pth.tar", model)
            else:
                state = load_saved_state(self.args, pretrain_level="1")
                model.load_state_dict(state['model'], strict=False)
        
        model = model.to(self.args.device)

        train_params = get_params(self.args, TrainingType.ACTIVE_LEARNING)
        optimizer, scheduler = load_optimizer(
            self.args, model.parameters(), 
            state, train_params,
            train_loader=train_loader)

        counter = 0
        for epoch in range(self.args.al_finetune_trainer_epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.args.al_finetune_trainer_epochs))
            logging.info('-' * 20)

            train_loss = self.train_finetuner(model, epoch, criterion, optimizer, scheduler, train_loader)
            epoch_acc, eval_loss = self.eval_finetuner(model, criterion, test_loader)

            # update learning rate
            if self.args.al_optimizer != "SwAV":
                scheduler.step()

            if epoch_acc <= self.best_trainer_acc:
                counter += 1
            else:
                counter = 0

            if counter > 20:
                logging.info("Early stopped at epoch {}:".format(epoch))
                break

        simple_save_model(self.args, self.best_model, f'{prefix}_finetuner_{self.dataset}.pth')

    def do_active_learning(self) -> List[PathLoss]:
        logging.info(f"Base = {get_dataset_enum(self.args.base_dataset)}, Target = {get_dataset_enum(self.args.target_dataset)}")
        encoder = resnet_backbone(self.args.backbone, pretrained=False)
        
        state = simple_load_model(self.args, path=f'first_finetuner_{self.dataset}.pth')
        if not state:
            self.finetuner(encoder, prefix='first')

        path_loss = load_path_loss(self.args, self.args.al_path_loss_file)
        if path_loss is None:
            path_loss = self.make_batches(encoder, prefix='first')

        return self.active_learning_new(path_loss, encoder)

    def active_learning_new(self, path_loss, encoder):
        pretraining_sample_pool = []

        gen_filename = f'generated_{get_dataset_enum(self.args.target_dataset)}'
        gen_images = glob.glob(f'{self.args.dataset_dir}/{gen_filename}/*')
        pretraining_gen_images = [PathLoss(path, 0) for path in gen_images]
        pretraining_sample_pool.extend(pretraining_gen_images)


        # # adding proxy images
        # source_proxy = glob.glob(f'{self.args.dataset_dir}/cifar10/train/*/*')
        # random.shuffle(source_proxy)
        # pretraining_source_proxy = [PathLoss(path, 0) for path in source_proxy]

        # augment_size = 3200
        # logging.info(f"Augmenting {augment_size} proxy source images to the generated dataset")
        # pretraining_sample_pool.extend(pretraining_source_proxy[0:augment_size])
        # # ###################

        logging.info(f"Size of pretraining_sample_pool is {len(pretraining_sample_pool)}")

        path_loss = path_loss[::-1] # this does a reverse active learning to pick only the most certain data
        logging.info(f"Size of the original data is {len(path_loss)}")

        if self.args.training_type in ['uc2', 'pete_2']:
            self.args.al_trainer_sample_size = int((len(path_loss))//self.args.al_batches)

        logging.info(f"Using a pretrain size of {self.args.al_trainer_sample_size} per AL batch.")

        sample_per_batch = len(path_loss)//self.args.al_batches

        batch_sampler_encoder = encoder

        for batch in range(self.args.al_batches):
            sample6400 = path_loss[batch * sample_per_batch : (batch + 1) * sample_per_batch]

            if batch > 0:
                logging.info(f'>> Getting best checkpoint for batch {batch + 1}')

                state = simple_load_model(self.args, path=f'{batch-1}_finetuner_{self.dataset}.pth')

                batch_sampler_encoder.load_state_dict(state['model'], strict=False)

                # sampling
                samplek = self.batch_sampler(batch_sampler_encoder, sample6400)[:self.args.al_trainer_sample_size]
                batch_sampler_encoder = encoder
            else:
                # first iteration: sample k at even intervals
                samplek = sample6400[:self.args.al_trainer_sample_size]

            pretraining_sample_pool.extend(samplek)

            logging.info(f"Size of pretraining_sample_pool is {len(pretraining_sample_pool)}")

            loader = PretextDataLoader(self.args, pretraining_sample_pool, training_type=TrainingType.BASE_PRETRAIN).get_loader()
            pretrainer = SelfSupPretrainer(self.args, self.writer)
            pretrainer.base_pretrain(encoder, loader, self.args.base_epochs, trainingType=TrainingType.BASE_PRETRAIN)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                self.finetuner_new(encoder, prefix=str(batch), path_list=pretraining_sample_pool, training_type=TrainingType.BASE_PRETRAIN)

        
        return pretraining_sample_pool