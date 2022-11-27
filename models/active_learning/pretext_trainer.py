import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.logger as logging
from typing import List

import random
import torchvision.transforms as transforms

from datautils.path_loss import PathLoss
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.pretext_dataloader import Loader, PretextDataLoader, RotationLoader
from models.backbones.resnet import resnet_backbone
from models.self_sup.myow.trainer.myow_trainer import get_myow_trainer
from models.self_sup.simclr.trainer.simclr_trainer import SimCLRTrainer
from models.self_sup.simclr.trainer.simclr_trainer_v2 import SimCLRTrainerV2

from models.utils.commons import get_model_criterion
from models.utils.training_type_enum import TrainingType
from models.utils.ssl_method_enum import SSL_Method
from models.active_learning.al_method_enum import AL_Method
from utils.commons import load_path_loss, load_saved_state, save_path_loss, simple_load_model, simple_save_model

class PretextTrainer():
    def __init__(self, args, writer) -> None:
        self.args = args
        self.writer = writer
        self.criterion = None
        self.log_step = 1000

    def train_proxy(self, samples, model, rebuild_al_model=False):

        # convert samples to loader
        loader = PretextDataLoader(self.args, samples).get_loader()
        logging.info("Beginning training the proxy")

        log_step = self.args.log_step
        if self.args.method == SSL_Method.SIMCLR.value:
            trainer = SimCLRTrainer(
                args=self.args, writer=self.writer, encoder=model, dataloader=loader, 
                pretrain_level="1", rebuild_al_model=rebuild_al_model, 
                training_type=TrainingType.ACTIVE_LEARNING, log_step=log_step)

        elif self.args.method == SSL_Method.DCL.value:
            trainer = SimCLRTrainerV2(
                args=self.args, writer=self.writer, encoder=model, dataloader=loader, 
                pretrain_level="1", rebuild_al_model=rebuild_al_model, 
                training_type=TrainingType.ACTIVE_LEARNING, log_step=log_step)

        elif self.args.method == SSL_Method.MYOW.value:
            trainer = get_myow_trainer(
                args=self.args, writer=self.writer, encoder=model, dataloader=loader, 
                pretrain_level="1", rebuild_al_model=rebuild_al_model, 
                trainingType=TrainingType.ACTIVE_LEARNING, log_step=log_step)

        else:
            ValueError

        for epoch in range(self.args.al_epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.args.al_epochs))
            logging.info('-' * 20)

            epoch_loss = trainer.train_epoch()

            # Decay Learning Rate
            trainer.scheduler.step()
            
            logging.info('Train Loss: {:.4f}'.format(epoch_loss))

        return trainer.model

    def train_proxy_(self, samples, model, rebuild_al_model=False):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ds = Loader(self.args, pathloss_list=samples, transform=transform_train, is_val=False)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

        if not rebuild_al_model:
            model.linear = nn.Linear(512, 4)
            state = load_saved_state(self.args, pretrain_level="1")
            model.load_state_dict(state['model'], strict=False)
            model = model.to(self.args.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

        model.train()
        total_loss, total_num = 0.0, 0

        for epoch in range(self.args.al_epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, self.args.al_epochs))
            logging.info('-' * 20)

            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()

                total_num += 128
                total_loss += loss.item() * 128

                if step % self.log_step == 0:
                    logging.info(f"Step [{step}/{len(self.train_loader)}]\t Loss: {total_loss / total_num}")

            # Decay Learning Rate
            scheduler.step()
            logging.info('Train Loss: {:.4f}'.format(total_loss))

        return model

    def finetune_(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ds = Loader(self.args, pathloss_list=samples, transform=transform_test, is_val=True)
        test_loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

        top1_scores = []
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = model(inputs)

                scores, predicted = outputs.max(1)
                # save top1 confidence score 

                outputs = F.normalize(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                top1_scores.append(probs[0][predicted.item()])

        print(top1_scores)
        idx = np.argsort(top1_scores)
        samples = np.array(samples)
        return samples[idx[:1000]]


    def finetune(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        # Train using 70% of the samples with the highest loss. So this should be the source of the data
        loader = PretextDataLoader(self.args, samples, training_type=TrainingType.AL_FINETUNING, is_val=True).get_loader()

        logging.info("Generating the top1 scores")
        _preds = []
        model.eval()

        with torch.no_grad():
            for step, (image, _) in enumerate(loader):
                # images[0] = images[0].to(self.args.device)
                # images[1] = images[1].to(self.args.device)

                # _, _, outputs1, outputs2 = model(images[0], images[1])
                image = image.to(self.args.device)

                if self.args.method == SSL_Method.SIMCLR.value:
                    features = model(image)
                
                else:
                    features, _ = model(image)

                _preds.append(self.get_preds(features))

                if step % self.args.log_step == 0:
                    logging.info(f"Step [{step}/{len(loader)}]")

        # preds = torch.cat(_preds).numpy()
       
        return self.get_new_samples_(_preds, samples)


    def get_predictions(self, outputs):
        dist1 = F.softmax(outputs, dim=1)
        preds = dist1.detach().cpu()

        return preds

    def get_preds(self, outputs):
        if self.args.al_method == AL_Method.LEAST_CONFIDENCE.value:
            _, predicted = outputs.max(1)
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            preds = probs[0][predicted.item()]

        else:
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            preds = e.view(e.size(0))

        return preds

    def get_new_samples_(self, preds, samples) -> List[PathLoss]:
        print(preds)
        if self.args.al_method == AL_Method.LEAST_CONFIDENCE.value:
            indices = np.argsort(preds)
            samples = np.array(samples)
            # return samples[indices[:1000]]
            indices = indices[:1000]

        elif self.args.al_method == AL_Method.ENTROPY.value:
            indices = np.argsort(preds)
            samples = np.array(samples)
            # return samples[indices[-1000:]]
            indices = indices[-1000:]

        print(indices)

        new_samples = []
        for item in indices:
            new_samples.append(samples[item]) # Map back to original indices

        return new_samples[:self.args.al_trainer_sample_size]

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

    def make_batches(self, encoder) -> List[PathLoss]:
        # This is a hack to the model can use a batch size of 1 to compute the loss for all the samples
        batch_size = self.args.al_batch_size
        self.args.al_batch_size = 1

        model = encoder
        model, criterion = get_model_criterion(self.args, model, training_type=TrainingType.ACTIVE_LEARNING, is_make_batches=True)
        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)

        model = model.to(self.args.device)
        loader = get_target_pretrain_ds(self.args, training_type=TrainingType.ACTIVE_LEARNING).get_loader()

        model.eval()
        pathloss = []

        logging.info("About to begin eval to make batches")
        count = 0
        with torch.no_grad():
            for step, (image, path) in enumerate(loader):
                image = image.to(self.args.device)

                # output = model(image)

                # # this needs to be really looked into
                # loss = criterion(output, output)

                # Forward pass to get output/logits
                output1 = model(image)
                output2 = model(image)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(output1, output2) + criterion(output2, output1)
                loss /= 4
                
                loss = loss.item()
                if step % self.args.log_step == 0:
                    logging.info(f"Step [{step}/{len(loader)}]\t Loss: {loss}")

                pathloss.append(PathLoss(path, loss))
                count +=1
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss, reverse=True)
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        self.args.al_batch_size = batch_size

        return sorted_samples

    def make_batches_(self, model):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = RotationLoader(self.args, dir="/cifar10v2", with_train=True, is_train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        model.linear = nn.Linear(512, 4)
        state = simple_load_model(self.args, path='finetuner.pth')
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        criterion = nn.CrossEntropyLoss()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        pathloss = []

        logging.info("About to begin eval to make batches")
        with torch.no_grad():
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
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
                    logging.info(f"Step [{step}/{len(testloader)}]\t Loss: {loss}")

                pathloss.append(PathLoss(path, loss))
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss, reverse=True)
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        return sorted_samples

    def finetune_trainer(self, model):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ds = RotationLoader(self.args, dir="/cifar10v2", with_train=True, is_train=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

        model.linear = nn.Linear(512, 4)
        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)
        model = model.to(self.args.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

        model.train()

        epochs = 5
        for epoch in range(epochs):
            logging.info('\nEpoch {}/{}'.format(epoch, epochs))
            logging.info('-' * 20)

            total_loss, total_num = 0, 0
            correct = 0
            total = 0
            for step, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(trainloader):
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

                total_loss += loss.item()
                total_num += 256

                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                if step % self.log_step == 0:
                    logging.info(f"Step [{step}/{len(self.train_loader)}]\t Loss: {total_loss / total_num}")

            scheduler.step()

        simple_save_model(self.args, model, 'finetuner.pth')


    def do_active_learning(self) -> List[PathLoss]:
        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        proxy_model = encoder

        state = simple_load_model(self.args, path='finetuner.pth')
        if not state:
            self.finetune_trainer(encoder)

        path_loss = load_path_loss(self.args, self.args.al_path_loss_file)
        if path_loss is None:
            path_loss = self.make_batches_(encoder)

        pretraining_sample_pool = []
        rebuild_al_model = True

        sample_per_batch = len(path_loss)//self.args.al_batches
        for batch in range(0, self.args.al_batches): # change the '1' to '0'
            sample6400 = path_loss[batch * sample_per_batch : (batch + 1) * sample_per_batch]

            # sketch -> 5120 | 2400 for 20 batches
            if batch > 0:
                logging.info(f'>> Getting previous checkpoint for batch {batch + 1}')

                state = simple_load_model(self.args, f'proxy_{batch-1}.pth')
                proxy_model.load_state_dict(state['model'], strict=False)

                # sampling
                samplek = self.finetune_(proxy_model, sample6400)
            else:
                # first iteration: sample k at even intervals
                samplek = sample6400[:self.args.al_trainer_sample_size]

            pretraining_sample_pool.extend(samplek)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                proxy_model = self.train_proxy_(
                    pretraining_sample_pool, 
                    proxy_model, rebuild_al_model=rebuild_al_model)

                rebuild_al_model=False
                simple_save_model(self.args, proxy_model, f'proxy_{batch}.pth')

        save_path_loss(self.args, self.args.pretrain_path_loss_file, pretraining_sample_pool)
        return pretraining_sample_pool