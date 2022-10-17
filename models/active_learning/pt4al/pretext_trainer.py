import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
import numpy as np

from datautils.image_loss_data import Image_Loss
from datautils.target_pretrain_dataset import get_target_pretrain_ds
from models.active_learning.pt4al.pretext_dataloader import PretextDataLoader
from models.backbones.resnet import resnet_backbone

from models.utils.commons import compute_loss, get_model_criterion
from utils.commons import load_saved_state, simple_load_model, simple_save_model

class PretextTrainer():
    def __init__(self, args, encoder) -> None:
        self.args = args
        self.loss_gen_loader = None
        self.finetune_loader = None
        self.encoder = encoder

        self.proxy_model, self.criterion = get_model_criterion(args, self.encoder)

    def train_proxy(self, samples, model, optimizer):

        # convert samples to loader
        loader = PretextDataLoader(self.args, samples).get_loader()

        model.train()
        for epoch in range(self.args.al_epochs):
            print("Proxy epoch - {}".format(epoch))

            for _, (images, _) in enumerate(loader):
                loss = compute_loss(self.args, images, model, self.criterion)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # Show progress here

    def finetune(self, model, samples):
        # Train using 70% of the samples with the highest loss. So this should be the source of the data
        loader = PretextDataLoader(self.args, samples, finetune=True).get_loader()

        top1_scores = []
        model.eval()
        with torch.no_grad():
            for _, (images, _) in enumerate(loader):
                images[0] = images[0].to(self.args.device)
                images[1] = images[1].to(self.args.device)

                outputs = model(images[0])
                scores, predicted = outputs.max(1)

                # save top1 confidence score 
                outputs = F.normalize(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                top1_scores.append(probs[0][predicted.item()])

        idx = np.argsort(top1_scores)
            
        # Save these images for use during the target pretraining
        return samples[idx[:2000]]

    def make_batches(self):
        model = self.encoder
        state = load_saved_state(self.args, pretrain_level="1")
        model.load_state_dict(state['model'])

        #@TODO remember to remove this and uncomment the lines above
        # encoder = resnet_backbone(self.args.al_backbone, pretrained=True)
        # model, self.criterion = get_model_criterion(self.args, encoder)

        model = model.to(self.args.device)
        loader = get_target_pretrain_ds(self.args).get_loader()

        model.eval()
        image_loss = []

        print("About to begin eval to make batches")
        with torch.no_grad():
            for _, (images, _) in enumerate(loader):
                loss = compute_loss(self.args, images, model, self.criterion)
                
                loss = loss.item()
                print(f"Loss: {loss}")

                image_loss.append(Image_Loss(images[0], images[1], loss))

        sorted_samples = sorted(image_loss, reverse=True)

        return sorted_samples

    def do_active_learning(self):
        self.proxy_model = self.proxy_model.to(self.args.device)

        optimizer = SGD(self.proxy_model.parameters(), lr=self.args.al_lr, momentum=0.9, weight_decay=self.args.al_weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        image_loss = self.make_batches()
        pretraining_sample_pool = []

        for batch in range(self.args.al_batches):
            sample6000 = image_loss[batch * 6000 : (batch + 1) * 6000]

            if batch > 0:
                print('>> Getting previous checkpoint')
                self.proxy_model.load_state_dict(simple_load_model(f'proxy_{batch-1}.pth'))

                # sampling
                sample2k = self.finetune(self.proxy_model, sample6000)
            else:
                # first iteration: sample 4k at even intervals
                sample2k = sample6000[2000:]

            pretraining_sample_pool.extend(sample2k)

            if batch < self.args.al_batches - 1: # I want this not to happen for the last iteration since it would be needless
                self.train_proxy(pretraining_sample_pool, self.proxy_model, optimizer)
                simple_save_model(self.args, self.proxy_model, f'proxy_{batch}.pth')
                scheduler.step()

        return pretraining_sample_pool

            
