import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
import numpy as np
from typing import Optional, Tuple, List

import random

from datautils.image_loss_data import Image_Loss
from datautils.imagenet import ImageNet
from datautils.path_loss import PathLoss
from datautils.target_dataset import get_target_pretrain_ds
from models.active_learning.method_enum import Method
from models.active_learning.pretext_dataloader import PretextDataLoader
from models.backbones.resnet import resnet_backbone

from models.utils.commons import compute_loss, get_model_criterion
from utils.commons import load_path_loss, load_saved_state, save_path_loss, simple_load_model, simple_save_model

class PretextTrainer():
    def __init__(self, args, encoder) -> None:
        self.args = args
        self.encoder = encoder
        self.criterion = None

    def train_proxy(self, samples, model, optimizer, scheduler=None):

        # convert samples to loader
        loader = PretextDataLoader(self.args, samples).get_loader(self.args.al_image_size)
        # loader = ImageNet(self.args, isAL=True).get_loader()
        print("Beginning training the proxy")

        model.train()
        for epoch in range(self.args.al_epochs):
            print(f"Epoch {epoch}")
            for step, (images) in enumerate(loader):
                loss = compute_loss(self.args, images, model, self.criterion)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if step % 5 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss}")

            # scheduler.step()


    def rough_finetune(self, outputs1, outputs2):
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)

        # save top1 confidence score 
        outputs1 = F.normalize(outputs1, dim=1)
        probs1 = F.softmax(outputs1, dim=1)
        print(predicted1.item())
        print(probs1[0])
        top1score1 = probs1[0][predicted1.item()]

        outputs2 = F.normalize(outputs2, dim=1)
        probs2 = F.softmax(outputs2, dim=1)
        top1score2 = probs2[0][predicted2.item()]
                                                    # Honestly not sure of what I am doing here, but the idea is since the two results belong
                                                    # to the two augmentations of the same image, so basically pick the largest score between the two :)
        # if top1score1 >= top1score2:
        #     top1_scores.append(top1score1)
        # else:
        #     top1_scores.append(top1score2)

                
    def finetune(self, model, samples: List[PathLoss]) -> List[PathLoss]:
        # Train using 70% of the samples with the highest loss. So this should be the source of the data
        loader = PretextDataLoader(self.args, samples, finetune=True).get_loader(self.args.al_image_size)

        model.eval()
        _preds = []
        print("Generating the top1 scores")
        with torch.no_grad():
            for step, (images) in enumerate(loader):
                images[0] = images[0].to(self.args.device)
                images[1] = images[1].to(self.args.device)

                _, _, outputs1, outputs2  = model(images[0], images[1])

                _preds.append(self.get_predictions(outputs1, outputs2))

                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]")

        preds = torch.cat(_preds).numpy()
       
        return self.get_new_samples(preds, samples)


    def get_predictions(self, outputs1, outputs2):
        dist1 = F.softmax(outputs1, dim=1)
        preds1 = dist1.detach().cpu()

        dist2 = F.softmax(outputs2, dim=1)
        preds2 = dist2.detach().cpu()

        # this is a hack since I don't know how else to handle it
        randnum = random.randint(0, 1)
        if randnum == 0:
            return preds1
        else:
            return preds2

    def get_new_samples(self, preds, samples) -> List[PathLoss]:
        if self.args.al_method == Method.LEAST_CONFIDENCE.value:
            probs = preds.max(axis=1)
            indices = probs.argsort(axis=0)

        elif self.args.al_method == Method.ENTROPY.value:
            entropy = (np.log(preds) * preds).sum(axis=1) * -1.
            indices = entropy.argsort(axis=0)[::-1]

        elif self.args.al_method == Method.BOTH.value:
            probs = preds.max(axis=1)
            indices1 = probs.argsort(axis=0)

            entropy = (np.log(preds) * preds).sum(axis=1) * -1.
            indices2 = entropy.argsort(axis=0)[::-1]

            indices = indices1 + indices2
            random.shuffle(indices)
            indices = indices[: (len(indices)/2)]

        else:
            raise NotImplementedError(f"'{self.args.al_method}' method doesn't exist")

        new_samples = []
        for item in indices:
            new_samples.append(samples[item]) # Map back to original indices

        return new_samples[:2000]

    def make_batches(self) -> List[PathLoss]:
        # This is a hack to the model can use a batch size of 1 to compute the loss for all the samples
        batch_size = self.args.al_batch_size
        self.args.al_batch_size = 1

        device = torch.device('cpu')

        model, criterion = get_model_criterion(self.args, self.encoder, isAL=True)
        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'], strict=False)

        #@TODO remember to remove this and uncomment the lines above
        # encoder = resnet_backbone(self.args.al_backbone, pretrained=True)
        # model, criterion = get_model_criterion(self.args, encoder, isAL=True)
        criterion = nn.CrossEntropyLoss().to(device)

        model = model.to(device)
        loader = get_target_pretrain_ds(self.args, isAL=True).get_loader()

        model.eval()
        pathloss = []

        print("About to begin eval to make batches")
        count = 0
        with torch.no_grad():
            for step, (images, path) in enumerate(loader):
                images[0] = images[0].to(device)
                images[1] = images[1].to(device)

                # positive pair, with encoding
                h_i, h_j, z_i, z_j = model(images[0], images[1])
                loss = criterion(z_i, z_j)
                
                loss = loss.item()
                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss}")

                pathloss.append(PathLoss(path, loss))
                count +=1
        
        sorted_samples = sorted(pathloss, key=lambda x: x.loss, reverse=True)
        save_path_loss(self.args, self.args.al_path_loss_file, sorted_samples)

        self.args.al_batch_size = batch_size

        return sorted_samples

    def do_active_learning(self) -> List[PathLoss]:
        path_loss = load_path_loss(self.args, self.args.al_path_loss_file)
        
        if path_loss is None:
            path_loss = self.make_batches()

        encoder = resnet_backbone(self.args.resnet, pretrained=False)
        proxy_model, self.criterion = get_model_criterion(self.args, encoder, isAL=True)
        proxy_model = proxy_model.to(self.args.device)

        optimizer = SGD(proxy_model.parameters(), lr=self.args.al_lr, momentum=0.9, weight_decay=self.args.al_weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        pretraining_sample_pool = []

        for batch in range(0, self.args.al_batches): # change the '1' to '0'
            sample6000 = path_loss[batch * 6000 : (batch + 1) * 6000] # this should be changed to a size of 6000

            if batch > 0:
                print('>> Getting previous checkpoint for batch ', batch + 1)
                proxy_model.load_state_dict(simple_load_model(self.args, f'proxy_{batch-1}.pth'))

                # sampling
                sample2k = self.finetune(proxy_model, sample6000)
            else:
                # first iteration: sample 4k at even intervals
                sample2k = sample6000[:2000] # this should be changed to a size of 2000

            pretraining_sample_pool.extend(sample2k)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                self.train_proxy(pretraining_sample_pool, proxy_model, optimizer, scheduler=None)
                simple_save_model(self.args, proxy_model, f'proxy_{batch}.pth')

        save_path_loss(self.args, self.args.pretrain_path_loss_file, pretraining_sample_pool)
        return pretraining_sample_pool

            
